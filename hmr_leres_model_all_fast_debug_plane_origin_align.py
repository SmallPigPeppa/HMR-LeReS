import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model import HMRNetBase
from Discriminator import Discriminator
from hmr_leres_config import args

from datasets.gta_im_all_10801920_plane_change_intri import GTADataset
from datasets.mesh_pkl_all import MeshDataset

from a_models.smpl_fixwarning import SMPL
from camera_utils import perspective_projection
from d_loss.hmr_loss import HMRLoss
from d_loss.leres_loss import DepthRegressionLoss, EdgeguidedNormalRegressionLoss, MultiScaleGradLoss, \
    recover_scale_shift_depth, EdgeguidedRankingLoss
# from d_loss.leres_loss_plane_debug_new_new import DepthRegressionLoss, NormalLoss
from d_loss.leres_loss_plane_debug_new import PWNPlanesLoss
from d_loss.align_loss_new_debug import AlignLoss

from lib_train.configs.config import cfg as leres_net_cfg

from a_models.leres import DepthModel
from datasets.hmr_data_utils import off_set_scale_kpts
from a_val_metrics.leres_metrics import val_depth
from a_val_metrics.hmr_metrics import val_kpts_verts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class HMRLeReS(pl.LightningModule):
    def __init__(self):
        super(HMRLeReS, self).__init__()
        self.automatic_optimization = False
        self.depth_min_threshold = 0.
        self.depth_max_threshold = 10.0
        self.pck_threshold = 1.0
        self.grad_clip_max_norm = 1.0
        self.hmr_generator = HMRNetBase()
        self.hmr_discriminator = Discriminator()
        self.smpl_model = SMPL(args.smpl_model, obj_saveable=True)
        self.leres_model = DepthModel()

        self.hmr_loss = HMRLoss()
        self.depth_regression_loss = DepthRegressionLoss(min_threshold=self.depth_min_threshold,
                                                         max_threshold=self.depth_max_threshold)
        # self.normal_loss = NormalLoss(min_threshold=self.depth_min_threshold,
        #                               max_threshold=self.depth_max_threshold, scale_factor=5.0)
        # plane
        # suface normal
        self.pwn_edge_loss = EdgeguidedNormalRegressionLoss(min_threshold=self.depth_min_threshold,
                                                            max_threshold=self.depth_max_threshold)
        self.pwn_plane_loss = PWNPlanesLoss(min_threshold=self.depth_min_threshold,
                                            max_threshold=self.depth_max_threshold,
                                            sample_groups=5000,
                                            input_size=(1080 // 5, 1920 // 5))
        #  可有可无
        # self.msg_loss = MultiScaleGradLoss(scale=4, min_threshold=self.depth_min_threshold,
        #                                    max_threshold=self.depth_max_threshold)
        #
        # self.edge_ranking_loss = EdgeguidedRankingLoss(min_threshold=self.depth_min_threshold,
        #                                                max_threshold=self.depth_max_threshold)
        self.align_loss = AlignLoss()

    def train_dataloader(self):
        gta_dataset_dir = os.path.join(args.gta_dataset_dir, 'FPS-5-test')
        mesh_dataset_dir = os.path.join(args.mesh_dataset_dir, 'FPS-30')

        gta_dataset = GTADataset(gta_dataset_dir)
        mesh_dataset = MeshDataset(mesh_dataset_dir)
        self.gta_dataset = gta_dataset

        # Calculate the size of the subset to be chosen from mesh_dataset
        subset_size = int(len(gta_dataset) * (args.adv_batch_size / args.batch_size))
        subset_indices = torch.randperm(len(mesh_dataset))[:subset_size]
        mesh_dataset_subset = torch.utils.data.Subset(mesh_dataset, subset_indices)

        gta_loader = DataLoader(
            dataset=gta_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers
        )
        mesh_loader = DataLoader(
            dataset=mesh_dataset_subset,
            batch_size=args.adv_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers
        )
        loaders = {'gta_loader': gta_loader, 'mesh_loader': mesh_loader}
        return loaders

    def val_dataloader(self):
        gta_dataset_dir = os.path.join(args.gta_dataset_dir, 'FPS-5-val')
        gta_dataset = GTADataset(gta_dataset_dir)
        self.gta_dataset = gta_dataset

        gta_loader = DataLoader(
            dataset=gta_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers
        )
        return gta_loader

    def get_smpl_kpts_verts(self, transl, pose, shape, focal_length):
        verts, kpts_3d, Rs = self.smpl_model(shape=shape, pose=pose, get_skin=True)
        batch_size = kpts_3d.shape[0]
        kpts_3d += transl.unsqueeze(dim=1)
        verts += transl.unsqueeze(dim=1)
        kpts_2d = perspective_projection(
            kpts_3d,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            # rotation=torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=kpts_3d.dtype, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=torch.zeros(size=[batch_size, 3], device=self.device),
            focal_length=focal_length,
            camera_center=torch.Tensor([384 / 2, 216 / 2]).to(self.device).expand(batch_size, -1)
        )

        return kpts_2d, kpts_3d, verts

    def training_step(self, batch, batch_index):
        gta_data = batch['gta_loader']
        mesh_data = batch['mesh_loader']
        hmr_images = gta_data['hmr_image']

        gt_smpl_theta = gta_data['theta']
        gt_smpl_shapes = gt_smpl_theta[:, 75:].contiguous()
        gt_smpl_poses = gt_smpl_theta[:, 3:75].contiguous()
        gt_smpl_transl = gt_smpl_theta[:, :3].contiguous()
        gt_kpts_2d = gta_data['kpts_2d_origin']
        gt_kpts_2d = gta_data['kpts_2d']
        gt_kpts_3d = gta_data['kpts_3d']
        gt_intrinsic = gta_data['intrinsic']
        gt_focal_length = gta_data['focal_length']
        top, left, height, width = gta_data['leres_cut_box'][:, 0], gta_data['leres_cut_box'][:, 1], \
            gta_data['leres_cut_box'][:, 2], gta_data['leres_cut_box'][:, 3]

        pred_smpl_thetas = self.hmr_generator(hmr_images)[-1]
        pred_smpl_transl = pred_smpl_thetas[:, :3].contiguous()
        pred_smpl_poses = pred_smpl_thetas[:, 3:75].contiguous()
        pred_smpl_shapes = pred_smpl_thetas[:, 75:].contiguous()

        pred_kpts_2d, pred_kpts_3d, pred_verts = self.get_smpl_kpts_verts(transl=gt_smpl_transl,
                                                                          pose=pred_smpl_poses,
                                                                          shape=pred_smpl_shapes,
                                                                          focal_length=gt_focal_length)
        # gt_kpts_2d, gt_kpts_3d, gt_verts = self.get_smpl_kpts_verts(transl=gt_smpl_transl,
        #                                           pose=gt_smpl_poses,
        #                                           shape=gt_smpl_shapes,
        #                                           focal_length=gt_focal_length)
        _,_, gt_verts = self.get_smpl_kpts_verts(transl=gt_smpl_transl,
                                                  pose=gt_smpl_poses,
                                                  shape=gt_smpl_shapes,
                                                  focal_length=gt_focal_length)

        # height_ratio = self.gta_dataset.leres_size[0] / height
        # width_ratio = self.gta_dataset.leres_size[1] / width
        # pred_kpts_2d[:, :, 0] -= left[:, None]
        # pred_kpts_2d[:, :, 1] -= top[:, None]
        # pred_kpts_2d[:, :, 0] *= width_ratio[:, None]
        # pred_kpts_2d[:, :, 1] *= height_ratio[:, None]

        # loss_shape = self.hmr_loss.shape_loss(gt_smpl_shapes, pred_smpl_shapes) * args.e_shape_weight
        loss_shape = self.hmr_loss.shape_loss(gt_smpl_shapes, pred_smpl_shapes) * 1.0
        # loss_pose = self.hmr_loss.pose_loss(gt_smpl_poses, pred_smpl_poses) * args.e_pose_weight
        loss_pose = self.hmr_loss.pose_loss(gt_smpl_poses, pred_smpl_poses) * 100.0
        # loss_kpts_2d = self.hmr_loss.batch_kp_2d_l1_loss(gt_kpts_2d, pred_kpts_2d) * args.e_2d_kpts_weight
        loss_kpts_2d = self.hmr_loss.batch_kp_2d_l1_loss(gt_kpts_2d, pred_kpts_2d) * 10.0
        # loss_kpts_2d = 0.

        # loss_kpts_3d = self.hmr_loss.batch_kp_3d_l2_loss(gt_kpts_3d, pred_kpts_3d) * args.e_3d_kpts_weight
        loss_kpts_3d = self.hmr_loss.batch_kp_3d_l2_loss(gt_kpts_3d, pred_kpts_3d) * 200.0

        pred_smpl_thetas[:, :3] = gt_smpl_transl
        loss_generator_disc = self.hmr_loss.batch_encoder_disc_l2_loss(
            self.hmr_discriminator(pred_smpl_thetas))

        real_thetas = mesh_data['theta']
        fake_thetas = pred_smpl_thetas.detach()
        fake_disc_value, real_disc_value = self.hmr_discriminator(fake_thetas), self.hmr_discriminator(real_thetas)
        d_disc_real, d_disc_fake, d_disc_loss = self.hmr_loss.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)

        loss_generator = (loss_shape + loss_pose + loss_kpts_2d + loss_kpts_3d) * args.e_loss_weight + \
                         loss_generator_disc * args.d_loss_weight

        loss_discriminator = d_disc_loss * args.d_loss_weight
        hmr_loss_dict = {'loss_generator': loss_generator,
                         'loss_kpts_2d': loss_kpts_2d,
                         'loss_kpts_3d': loss_kpts_3d,
                         'loss_shape': loss_shape,
                         'loss_pose': loss_pose,
                         'loss_generator_disc': loss_generator_disc,
                         'loss_discriminator': loss_discriminator,
                         'd_disc_real': d_disc_real,
                         'd_disc_fake': d_disc_fake
                         }

        leres_images = gta_data['leres_image']
        pred_depth, auxi = self.leres_model(leres_images)
        gt_depth = gta_data['depth']
        gt_depth = gt_depth[:, None, :, :]
        plane_mask = gta_data['plane_mask']

        loss_depth_regression = self.depth_regression_loss(torch.squeeze(pred_depth), torch.squeeze(gt_depth))
        # loss_normal = self.normal_loss(torch.squeeze(pred_depth), torch.squeeze(gt_depth), gt_intrinsic)
        loss_normal = 0.
        # loss_depth_regression = 0.
        # loss_edge_ranking = self.edge_ranking_loss(pred_depth, gt_depth, leres_images)
        loss_edge_ranking = 0.
        # loss_msg = self.msg_loss(pred_depth, gt_depth) * 0.5
        loss_msg = 0.
        # pred_ssinv = recover_scale_shift_depth(pred_depth, gt_depth, min_threshold=0., max_threshold=15.0)
        # pred_ssinv = 0.
        # loss_pwn_edge = self.pwn_edge_loss(pred_ssinv, gt_depth, leres_images, focal_length=gt_focal_length)
        loss_pwn_edge = self.pwn_edge_loss(pred_depth, gt_depth, leres_images, focal_length=gt_focal_length)
        # loss_pwn_edge = 0.
        loss_pwn_plane = self.pwn_plane_loss(pred_depth, gt_depth, plane_mask, focal_length=gt_focal_length)
        # loss_pwn_plane = 0.

        loss_leres = (
                loss_depth_regression + loss_edge_ranking + loss_msg + loss_pwn_edge + loss_pwn_plane + loss_normal * 10)
        # loss_leres = 0.

        leres_loss_dict = {
            'loss_depth_regression': loss_depth_regression,
            'loss_normal': loss_normal,
            'loss_edge_ranking': loss_edge_ranking,
            'loss_msg': loss_msg,
            'loss_pwn_edge': loss_pwn_edge,
            'loss_pwn_plane': loss_pwn_plane,
            'loss_leres': loss_leres,
        }

        # loss_align
        # loss_align = self.align_loss.batch_align_loss(pred_verts,
        #                                               torch.tensor([self.smpl_model.faces], device=self.device),
        #                                               pred_depth, gta_data)
        # loss_align = 0.
        faces = torch.tensor([self.smpl_model.faces], device=self.device)
        loss_align = self.align_loss(faces, gt_verts, pred_verts, pred_depth, gta_data)
        human_depth_loss = loss_align['human_depth_loss']
        mesh_project_loss = loss_align['mesh_project_loss']
        mesh_project_edge_loss = loss_align['mesh_project_edge_loss']
        loss_combie = human_depth_loss + mesh_project_loss + mesh_project_edge_loss

        # loss_combie = 0.
        combine_loss_dict = {
            'human_depth_loss': human_depth_loss,
            'mesh_project_loss': mesh_project_loss,
            'mesh_project_edge_loss': mesh_project_edge_loss,
            'loss_combine': loss_combie
        }

        # depths_metrics = val_depth(pred_depth, gt_depth, min_threshold=self.depth_min_threshold,
        #                            max_threshold=self.depth_max_threshold)
        depths_metrics = {}

        kpts_verts_metrics = val_kpts_verts(pred_kpts_3d, gt_kpts_3d, pred_verts, gt_verts,
                                            pck_threshold=self.pck_threshold)

        log_dict = {**leres_loss_dict, **hmr_loss_dict, **combine_loss_dict, **kpts_verts_metrics, **depths_metrics}

        hmr_generator_leres_opt, hmr_discriminator_opt = self.optimizers()

        # hmr_generator and leres_model
        hmr_generator_leres_opt.zero_grad()
        self.manual_backward(log_dict['loss_generator'] + log_dict['loss_leres'] + log_dict['loss_combine'])
        torch.nn.utils.clip_grad_norm_(self.hmr_generator.parameters(), max_norm=self.grad_clip_max_norm)
        torch.nn.utils.clip_grad_norm_(self.leres_model.parameters(), max_norm=self.grad_clip_max_norm)
        hmr_generator_leres_opt.step()

        # hmr_discriminator
        hmr_discriminator_opt.zero_grad()
        self.manual_backward(log_dict['loss_discriminator'])
        torch.nn.utils.clip_grad_norm_(self.hmr_discriminator.parameters(), max_norm=self.grad_clip_max_norm)
        hmr_discriminator_opt.step()

        train_log_dict = {f'train_{k}': v for k, v in log_dict.items()}
        self.log_dict(train_log_dict, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_outputs):
        hmr_generator_leres_sche, hmr_discriminator_sche = self.lr_schedulers()
        hmr_generator_leres_sche.step()
        hmr_discriminator_sche.step()

    def configure_optimizers(self):
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

        hmr_generator_leres_params = [
            {'params': self.hmr_generator.parameters(),
             'lr': args.e_lr,
             'weight_decay': args.e_wd},
            {'params': leres_encoder_params,
             'lr': args.base_lr,
             'weight_decay': args.weight_decay},
            {'params': leres_decoder_params,
             'lr': args.base_lr * args.scale_decoder_lr,
             'weight_decay': args.weight_decay}
        ]

        hmr_discriminator_params = [
            {'params': self.hmr_discriminator.parameters(),
             'lr': args.d_lr,
             'weight_decay': args.d_wd}]

        hmr_generator_leres_opt = torch.optim.SGD(
            hmr_generator_leres_params, momentum=0.9
        )

        hmr_discriminator_opt = torch.optim.Adam(
            hmr_discriminator_params
        )

        hmr_generator_lere_sche = LinearWarmupCosineAnnealingLR(
            hmr_generator_leres_opt,
            warmup_epochs=5,
            max_epochs=args.max_epochs,
            warmup_start_lr=0.01 * args.e_lr,
            eta_min=0.01 * args.e_lr,
        )

        hmr_discriminator_sche = LinearWarmupCosineAnnealingLR(
            hmr_discriminator_opt,
            warmup_epochs=5,
            max_epochs=args.max_epochs,
            warmup_start_lr=0.01 * args.d_lr,
            eta_min=0.01 * args.d_lr,
        )

        return [hmr_generator_leres_opt, hmr_discriminator_opt], [hmr_generator_lere_sche, hmr_discriminator_sche]

    # def validation_step(self, batch, batch_idx):
    #     gta_data = batch
    #     hmr_images = gta_data['hmr_image']
    #
    #     gt_smpl_theta = gta_data['theta']
    #     gt_smpl_shapes = gt_smpl_theta[:, 75:].contiguous()
    #     gt_smpl_poses = gt_smpl_theta[:, 3:75].contiguous()
    #     gt_smpl_transl = gt_smpl_theta[:, :3].contiguous()
    #     gt_kpts_2d = gta_data['kpts_2d']
    #     gt_kpts_3d = gta_data['kpts_3d']
    #     gt_intrinsic = gta_data['intrinsic']
    #     gt_focal_length = gta_data['focal_length']
    #     top, left, height, width = gta_data['leres_cut_box'][:, 0], gta_data['leres_cut_box'][:, 1], \
    #                                gta_data['leres_cut_box'][:, 2], gta_data['leres_cut_box'][:, 3]
    #
    #     pred_smpl_thetas = self.hmr_generator(hmr_images)[-1]
    #     pred_smpl_transl = pred_smpl_thetas[:, :3].contiguous()
    #     pred_smpl_poses = pred_smpl_thetas[:, 3:75].contiguous()
    #     pred_smpl_shapes = pred_smpl_thetas[:, 75:].contiguous()
    #
    #     pred_kpts_2d, pred_kpts_3d, pred_verts = self.get_smpl_kpts_verts(transl=gt_smpl_transl,
    #                                                                       pose=pred_smpl_poses,
    #                                                                       shape=pred_smpl_shapes,
    #                                                                       focal_length=gt_focal_length)
    #     _, _, gt_verts = self.get_smpl_kpts_verts(transl=gt_smpl_transl,
    #                                               pose=gt_smpl_poses,
    #                                               shape=gt_smpl_shapes,
    #                                               focal_length=gt_focal_length)
    #
    #     height_ratio = self.gta_dataset.leres_size / height
    #     width_ratio = self.gta_dataset.leres_size / width
    #     pred_kpts_2d[:, :, 0] -= left[:, None]
    #     pred_kpts_2d[:, :, 1] -= top[:, None]
    #     pred_kpts_2d[:, :, 0] *= height_ratio[:, None]
    #     pred_kpts_2d[:, :, 1] *= width_ratio[:, None]
    #
    #     loss_shape = self.hmr_loss.shape_loss(gt_smpl_shapes, pred_smpl_shapes) * args.e_shape_weight
    #     loss_pose = self.hmr_loss.pose_loss(gt_smpl_poses, pred_smpl_poses) * args.e_pose_weight
    #     # import pdb;pdb.set_trace()
    #     print(gt_kpts_2d.shape,pred_kpts_2d.shape)
    #     loss_kpts_2d = self.hmr_loss.batch_kp_2d_l1_loss(gt_kpts_2d, pred_kpts_2d) * args.e_2d_kpts_weight
    #
    #     # loss_kpts_2d = 0.
    #
    #     loss_kpts_3d = self.hmr_loss.batch_kp_3d_l2_loss(gt_kpts_3d, pred_kpts_3d) * args.e_3d_kpts_weight
    #
    #     pred_smpl_thetas[:, :3] = gt_smpl_transl
    #     loss_generator_disc = self.hmr_loss.batch_encoder_disc_l2_loss(
    #         self.hmr_discriminator(pred_smpl_thetas))
    #
    #     loss_generator = (loss_shape + loss_pose + loss_kpts_2d + loss_kpts_3d) * args.e_loss_weight + \
    #                      loss_generator_disc * args.d_loss_weight
    #
    #     hmr_loss_dict = {'loss_generator': loss_generator,
    #                      'loss_kpts_2d': loss_kpts_2d,
    #                      'loss_kpts_3d': loss_kpts_3d,
    #                      'loss_shape': loss_shape,
    #                      'loss_pose': loss_pose,
    #                      'loss_generator_disc': loss_generator_disc,
    #                      }
    #
    #     leres_images = gta_data['leres_image']
    #     predict_depth, auxi = self.leres_model(leres_images)
    #     gt_depth = gta_data['depth']
    #     gt_depth = gt_depth[:, None, :, :]
    #     loss_depth_regression = self.depth_regression_loss(predict_depth, gt_depth)
    #     loss_edge_ranking = self.edge_ranking_loss(predict_depth, gt_depth, leres_images)
    #     loss_msg = self.msg_loss(predict_depth, gt_depth) * 0.5
    #     pred_ssinv = recover_scale_shift_depth(predict_depth, gt_depth, min_threshold=0., max_threshold=15.0)
    #     loss_pwn_edge = self.pwn_edge_loss(pred_ssinv, gt_depth, leres_images, focal_length=gt_focal_length)
    #     loss_leres = (loss_depth_regression + loss_edge_ranking + loss_msg + loss_pwn_edge)
    #     leres_loss_dict = {
    #         'loss_depth_regression': loss_depth_regression,
    #         'loss_edge_ranking': loss_edge_ranking,
    #         'loss_msg': loss_msg,
    #         'loss_pwn_edge': loss_pwn_edge,
    #         'loss_leres': loss_leres
    #     }
    #
    #     # loss_align
    #     loss_align = self.align_loss.batch_align_loss(pred_verts,
    #                                                   torch.tensor([self.smpl_model.faces], device=self.device),
    #                                                   predict_depth, gta_data)
    #     loss_inside = 0.
    #     loss_combie = loss_align + loss_inside
    #     combine_loss_dict = {
    #         'loss_align': loss_align,
    #         'loss_inside': loss_inside,
    #         'loss_combine': loss_combie
    #     }
    #
    #     depths_metrics = val_depth(predict_depth, gt_depth, min_threshold=self.depth_min_threshold,
    #                                max_threshold=self.depth_max_threshold)
    #     kpts_verts_metrics = val_kpts_verts(pred_kpts_3d, gt_kpts_3d, pred_verts, gt_verts,
    #                                         pck_threshold=self.pck_threshold)
    #
    #     log_dict = {**leres_loss_dict, **hmr_loss_dict, **combine_loss_dict, **kpts_verts_metrics, **depths_metrics}
    #
    #     save_ckpt_loss = log_dict['loss_generator'] + log_dict['loss_leres'] + log_dict['loss_combine']
    #     val_log_dict = {f'val_{k}': v for k, v in log_dict.items()}
    #     self.log_dict({**val_log_dict, 'save_ckpt_loss': save_ckpt_loss})
