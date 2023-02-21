import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model import HMRNetBase
from Discriminator import Discriminator
from hmr_leres_config import args

from datasets.gta_im import GTADataset
from datasets.mesh_pkl import MeshDataset

from a_models.smpl import SMPL
from camera_utils import perspective_projection
from d_loss.hmr_loss import HMRLoss
from d_loss.leres_loss import DepthRegressionLoss, EdgeguidedNormalRegressionLoss, MultiScaleGradLoss, \
    recover_scale_shift_depth, EdgeguidedRankingLoss

from lib_train.configs.config import cfg as leres_net_cfg

from a_models.leres import DepthModel
from datasets.hmr_data_utils import off_set_scale_kpts

class HMRLeReS(pl.LightningModule):
    def __init__(self):
        super(HMRLeReS, self).__init__()
        self.automatic_optimization = False
        self.hmr_generator = HMRNetBase()
        self.hmr_discriminator = Discriminator()
        self.smpl_model = SMPL(args.smpl_model, obj_saveable=True)
        self.hmr_loss = HMRLoss()
        self.leres_model = DepthModel()

        self.depth_regression_loss = DepthRegressionLoss(min_threshold=0., max_threshold=15)
        self.pwn_edge_loss = EdgeguidedNormalRegressionLoss(min_threshold=0., max_threshold=15)
        self.msg_loss = MultiScaleGradLoss(scale=4, min_threshold=0., max_threshold=15)
        self.edge_ranking_loss = EdgeguidedRankingLoss(min_threshold=0., max_threshold=15)

    def train_dataloader(self):
        gta_dataset_dir = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48-add-transl'
        mesh_dataset_dir = 'C:/Users/90532/Desktop/Datasets/HMR-LeRes/mesh-add-transl'

        # gta_dataset_dir = '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48'
        # mesh_dataset_dir = '/share/wenzhuoliu/torch_ds/HMR-LeReS/mosh'

        gta_dataset = GTADataset(gta_dataset_dir)
        mesh_dataset = MeshDataset(mesh_dataset_dir)
        self.gta_dataset=gta_dataset

        gta_loader = DataLoader(
            dataset=gta_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_worker
        )

        mesh_loader = DataLoader(
            dataset=mesh_dataset,
            batch_size=args.adv_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        loaders = {'gta_loader': gta_loader, 'mesh_loader': mesh_loader}
        return loaders

    def get_smpl_kpts(self, transl, pose, shape, focal_length):
        verts, kpts_3d, Rs = self.smpl_model(shape=shape, pose=pose, get_skin=True)
        batch_size = kpts_3d.shape[0]
        kpts_3d += transl.unsqueeze(dim=1)
        kpts_2d = perspective_projection(
            kpts_3d,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=torch.zeros(size=[batch_size, 3], device=self.device),
            focal_length=focal_length,
            camera_center=torch.Tensor([960.0, 540.0]).to(self.device).expand(batch_size, -1)
        )

        return kpts_2d, kpts_3d

    def training_step(self, batch, batch_index):
        gta_data = batch['gta_loader']
        mesh_data = batch['mesh_loader']
        hmr_images = gta_data['hmr_image']

        gt_smpl_theta = gta_data['theta']
        gt_smpl_shapes = gt_smpl_theta[:, 75:].contiguous()
        gt_smpl_poses = gt_smpl_theta[:, 3:75].contiguous()
        gt_smpl_transl = gt_smpl_theta[:, :3].contiguous()
        gt_kpts_2d = gta_data['joints_2d_origin']
        gt_kpts_3d = gta_data['joints_3d']
        gt_intrinsic = gta_data['intrinsic']
        gt_focal_length = gta_data['focal_length']
        top, left, height, width=gta_data['leres_cut_box'][:,0],gta_data['leres_cut_box'][:,1],gta_data['leres_cut_box'][:,2],gta_data['leres_cut_box'][:,3]

        predict_smpl_thetas = self.hmr_generator(hmr_images)[-1]
        predict_smpl_transl = predict_smpl_thetas[:, :3].contiguous()
        predict_smpl_poses = predict_smpl_thetas[:, 3:75].contiguous()
        predict_smpl_shapes = predict_smpl_thetas[:, 75:].contiguous()

        predict_kpts_2d, predict_kpts_3d = self.get_smpl_kpts(transl=gt_smpl_transl, pose=predict_smpl_poses,
                                                              shape=predict_smpl_shapes, focal_length=gt_focal_length)

        height_ratio = self.gta_dataset.leres_size / height
        width_ratio = self.gta_dataset.leres_size / width
        predict_kpts_2d[:,:,0]-=left[:,None]
        predict_kpts_2d[:,:,1]-=top[:,None]
        predict_kpts_2d[:, :, 0] *= height_ratio[:,None]
        predict_kpts_2d[:, :, 1] *= width_ratio[:,None]


        loss_shape = self.hmr_loss.shape_loss(gt_smpl_shapes, predict_smpl_shapes) * args.e_shape_weight
        loss_pose = self.hmr_loss.pose_loss(gt_smpl_poses, predict_smpl_poses) * args.e_pose_weight
        loss_kpts_2d = self.hmr_loss.batch_kp_2d_l1_loss(gt_kpts_2d, predict_kpts_2d) * args.e_2d_kpts_weight
        # loss_kpts_2d = 0.

        loss_kpts_3d = self.hmr_loss.batch_kp_3d_l2_loss(gt_kpts_3d, predict_kpts_3d) * args.e_3d_kpts_weight

        predict_smpl_thetas[:, :3] = gt_smpl_transl
        loss_generator_disc = self.hmr_loss.batch_encoder_disc_l2_loss(
            self.hmr_discriminator(predict_smpl_thetas))

        real_thetas = mesh_data['theta']
        fake_thetas = predict_smpl_thetas.detach()
        fake_disc_value, real_disc_value = self.hmr_discriminator(fake_thetas), self.hmr_discriminator(real_thetas)
        d_disc_real, d_disc_fake, d_disc_loss = self.hmr_loss.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)

        # loss_generator = (loss_shape + loss_pose + loss_kpts_2d + loss_kpts_3d) * args.e_loss_weight + \
        #                  loss_generator_disc * args.d_loss_weight

        loss_generator = (loss_shape + loss_pose + loss_kpts_2d + loss_kpts_3d) * args.e_loss_weight + \
                         loss_generator_disc * args.d_loss_weight

        loss_discriminator = d_disc_loss * args.d_loss_weight

        leres_images = gta_data['leres_image']
        predict_depth, auxi = self.leres_model(leres_images)
        gt_depth = gta_data['depth']
        gt_depth = gt_depth[:, None, :, :]
        loss_depth_regression = self.depth_regression_loss(predict_depth, gt_depth)
        loss_edge_ranking = self.edge_ranking_loss(predict_depth, gt_depth, leres_images)
        loss_msg = self.msg_loss(predict_depth, gt_depth) * 0.5
        pred_ssinv = recover_scale_shift_depth(predict_depth, gt_depth, min_threshold=0., max_threshold=15.0)
        loss_pwn_edge = self.pwn_edge_loss(pred_ssinv, gt_depth, leres_images, focal_length=gt_focal_length)
        loss_leres = (loss_depth_regression + loss_edge_ranking + loss_msg + loss_pwn_edge)
        leres_log_dict = {
            'loss_depth_regression': loss_depth_regression,
            'loss_edge_ranking': loss_edge_ranking,
            'loss_msg': loss_msg,
            'loss_pwn_edge': loss_pwn_edge,
            'loss_leres': loss_leres
        }

        hmr_generator_opt, hmr_discriminator_opt, leres_opt = self.optimizers()

        leres_opt.zero_grad()
        self.manual_backward(loss_leres)
        torch.nn.utils.clip_grad_norm_(self.leres_model.parameters(), max_norm=10.0)
        leres_opt.step()

        hmr_generator_opt.zero_grad()
        self.manual_backward(loss_generator)
        hmr_generator_opt.step()

        hmr_discriminator_opt.zero_grad()
        self.manual_backward(loss_discriminator)
        hmr_discriminator_opt.step()

        hmr_log_dict = {'loss_generator': loss_generator,
                        'loss_kpts_2d': loss_kpts_2d,
                        'loss_kpts_3d': loss_kpts_3d,
                        'loss_shape': loss_shape,
                        'loss_pose': loss_pose,
                        'loss_generator_disc': loss_generator_disc,
                        'loss_discriminator': loss_discriminator,
                        'd_disc_real': d_disc_real,
                        'd_disc_fake': d_disc_fake
                        }

        all_log_dict = {**leres_log_dict,**hmr_log_dict}
        self.log_dict(all_log_dict)

    def training_epoch_end(self, training_step_outputs):
        hmr_generator_sche, hmr_discriminator_sche = self.lr_schedulers()
        hmr_generator_sche.step()
        hmr_discriminator_sche.step()

    def configure_optimizers(self):
        hmr_generator_opt = torch.optim.Adam(
            self.hmr_generator.parameters(),
            lr=args.e_lr,
            weight_decay=args.e_wd
        )
        hmr_discriminator_opt = torch.optim.Adam(
            self.hmr_discriminator.parameters(),
            lr=args.d_lr,
            weight_decay=args.d_wd
        )
        hmr_generator_sche = torch.optim.lr_scheduler.StepLR(
            hmr_generator_opt,
            step_size=500,
            gamma=0.9
        )
        hmr_discriminator_sche = torch.optim.lr_scheduler.StepLR(
            hmr_discriminator_opt,
            step_size=500,
            gamma=0.9
        )

        leres_encoder_params = []
        leres_encoder_params_names = []
        leres_decoder_params = []
        leres_decoder_params_names = []
        leres_nograd_param_names = []
        for key, value in self.named_parameters():
            if 'leres_model' in key and value.requires_grad:
                if 'res' in key:
                    leres_encoder_params.append(value)
                    leres_encoder_params_names.append(key)
                else:
                    leres_decoder_params.append(value)
                    leres_decoder_params_names.append(key)
            else:
                leres_nograd_param_names.append(key)

        leres_lr_encoder = leres_net_cfg.TRAIN.BASE_LR
        leres_lr_decoder = leres_net_cfg.TRAIN.BASE_LR * leres_net_cfg.TRAIN.SCALE_DECODER_LR
        leres_weight_decay = 0.0005

        leres_net_params = [
            {'params': leres_encoder_params,
             'lr': leres_lr_encoder,
             'weight_decay': leres_weight_decay},
            {'params': leres_decoder_params,
             'lr': leres_lr_decoder,
             'weight_decay': leres_weight_decay},
        ]
        leres_opt = torch.optim.SGD(leres_net_params, momentum=0.9)

        return [hmr_generator_opt, hmr_discriminator_opt, leres_opt], [hmr_generator_sche, hmr_discriminator_sche]
