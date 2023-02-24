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
from d_loss.align_loss import AlignLoss

from lib_train.configs.config import cfg as leres_net_cfg

from a_models.leres import DepthModel
from datasets.hmr_data_utils import off_set_scale_kpts
from a_val_metrics.leres_metrics import val_depth
from a_val_metrics.hmr_metrics import val_kpts_verts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.trainer.suppporters import CombinedLoader


class HMRLeReS(pl.LightningModule):
    def __init__(self):
        super(HMRLeReS, self).__init__()
        self.automatic_optimization = False
        self.depth_min_threshold = 0.
        self.depth_max_threshold = 15.0
        self.pck_threshold = 1.0
        self.hmr_generator = HMRNetBase()
        self.hmr_discriminator = Discriminator()
        self.smpl_model = SMPL(args.smpl_model, obj_saveable=True)
        self.leres_model = DepthModel()

        self.hmr_loss = HMRLoss()
        self.depth_regression_loss = DepthRegressionLoss(min_threshold=self.depth_min_threshold,
                                                         max_threshold=self.depth_max_threshold)
        self.pwn_edge_loss = EdgeguidedNormalRegressionLoss(min_threshold=self.depth_min_threshold,
                                                            max_threshold=self.depth_max_threshold)
        self.msg_loss = MultiScaleGradLoss(scale=4, min_threshold=self.depth_min_threshold,
                                           max_threshold=self.depth_max_threshold)
        self.edge_ranking_loss = EdgeguidedRankingLoss(min_threshold=self.depth_min_threshold,
                                                       max_threshold=self.depth_max_threshold)
        self.align_loss = AlignLoss()

    def train_dataloader(self):
        gta_dataset_dir = args.gta_dataset_dir
        mesh_dataset_dir = args.mesh_dataset_dir

        gta_dataset = GTADataset(gta_dataset_dir)
        mesh_dataset = MeshDataset(mesh_dataset_dir)
        self.gta_dataset = gta_dataset

        gta_loader = DataLoader(
            dataset=gta_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers
        )

        mesh_loader = DataLoader(
            dataset=mesh_dataset,
            batch_size=args.adv_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers
        )
        loaders = {'gta_loader': gta_loader, 'mesh_loader': mesh_loader}
        return loaders

    def val_dataloader(self):
        gta_dataset_dir = args.gta_dataset_dir
        mesh_dataset_dir = args.mesh_dataset_dir

        gta_dataset = GTADataset(gta_dataset_dir)
        mesh_dataset = MeshDataset(mesh_dataset_dir)
        self.gta_dataset = gta_dataset

        gta_loader = DataLoader(
            dataset=gta_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers
        )

        mesh_loader = DataLoader(
            dataset=mesh_dataset,
            batch_size=args.adv_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers
        )
        loaders = {'gta_loader': gta_loader, 'mesh_loader': mesh_loader}
        return CombinedLoader(loaders)

    def get_smpl_kpts_verts(self, transl, pose, shape, focal_length):
        verts, kpts_3d, Rs = self.smpl_model(shape=shape, pose=pose, get_skin=True)
        batch_size = kpts_3d.shape[0]
        kpts_3d += transl.unsqueeze(dim=1)
        verts += transl.unsqueeze(dim=1)
        kpts_2d = perspective_projection(
            kpts_3d,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=torch.zeros(size=[batch_size, 3], device=self.device),
            focal_length=focal_length,
            camera_center=torch.Tensor([960.0, 540.0]).to(self.device).expand(batch_size, -1)
        )

        return kpts_2d, kpts_3d, verts

    def share_step(self, batch):
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
        _, _, gt_verts = self.get_smpl_kpts_verts(transl=gt_smpl_transl,
                                                  pose=gt_smpl_poses,
                                                  shape=gt_smpl_shapes,
                                                  focal_length=gt_focal_length)

        height_ratio = self.gta_dataset.leres_size / height
        width_ratio = self.gta_dataset.leres_size / width
        pred_kpts_2d[:, :, 0] -= left[:, None]
        pred_kpts_2d[:, :, 1] -= top[:, None]
        pred_kpts_2d[:, :, 0] *= height_ratio[:, None]
        pred_kpts_2d[:, :, 1] *= width_ratio[:, None]

        loss_shape = self.hmr_loss.shape_loss(gt_smpl_shapes, pred_smpl_shapes) * args.e_shape_weight
        loss_pose = self.hmr_loss.pose_loss(gt_smpl_poses, pred_smpl_poses) * args.e_pose_weight
        loss_kpts_2d = self.hmr_loss.batch_kp_2d_l1_loss(gt_kpts_2d, pred_kpts_2d) * args.e_2d_kpts_weight
        # loss_kpts_2d = 0.

        loss_kpts_3d = self.hmr_loss.batch_kp_3d_l2_loss(gt_kpts_3d, pred_kpts_3d) * args.e_3d_kpts_weight

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
        predict_depth, auxi = self.leres_model(leres_images)
        gt_depth = gta_data['depth']
        gt_depth = gt_depth[:, None, :, :]
        loss_depth_regression = self.depth_regression_loss(predict_depth, gt_depth)
        loss_edge_ranking = self.edge_ranking_loss(predict_depth, gt_depth, leres_images)
        loss_msg = self.msg_loss(predict_depth, gt_depth) * 0.5
        pred_ssinv = recover_scale_shift_depth(predict_depth, gt_depth, min_threshold=0., max_threshold=15.0)
        loss_pwn_edge = self.pwn_edge_loss(pred_ssinv, gt_depth, leres_images, focal_length=gt_focal_length)
        loss_leres = (loss_depth_regression + loss_edge_ranking + loss_msg + loss_pwn_edge)
        leres_loss_dict = {
            'loss_depth_regression': loss_depth_regression,
            'loss_edge_ranking': loss_edge_ranking,
            'loss_msg': loss_msg,
            'loss_pwn_edge': loss_pwn_edge,
            'loss_leres': loss_leres
        }

        # loss_align
        loss_align = self.align_loss.batch_align_loss(pred_verts,
                                                      torch.tensor([self.smpl_model.faces], device=self.device),
                                                      predict_depth, gta_data)
        loss_inside = 0.
        loss_combie = loss_align + loss_inside
        combine_loss_dict = {
            'loss_align': loss_align,
            'loss_inside': loss_inside,
            'loss_combie': loss_combie
        }

        depths_metrics = val_depth(predict_depth, gt_depth, min_threshold=self.depth_min_threshold,
                                   max_threshold=self.depth_max_threshold)
        kpts_verts_metrics = val_kpts_verts(pred_kpts_3d, gt_kpts_3d, pred_verts, gt_verts,
                                            pck_threshold=self.pck_threshold)

        all_log_dict = {**leres_loss_dict, **hmr_loss_dict, **combine_loss_dict, **kpts_verts_metrics, **depths_metrics}

        return all_log_dict


    def training_step(self, batch, batch_index):
        log_dict = self.share_step(batch)
        hmr_generator_leres_opt, hmr_discriminator_opt = self.optimizers()

        # hmr_generator and leres_model
        hmr_generator_leres_opt.zero_grad()
        self.manual_backward(log_dict['loss_generator'] + log_dict['loss_leres'] + log_dict['loss_combine'])
        torch.nn.utils.clip_grad_norm_(self.hmr_generator.parameters(), max_norm=3.0)
        torch.nn.utils.clip_grad_norm_(self.leres_model.parameters(), max_norm=3.0)
        hmr_generator_leres_opt.step()

        # hmr_discriminator
        hmr_discriminator_opt.zero_grad()
        self.manual_backward(log_dict['loss_discriminator'])
        torch.nn.utils.clip_grad_norm_(self.hmr_discriminator.parameters(), max_norm=3.0)
        hmr_discriminator_opt.step()

        train_log_dict = {f'train_{k}': v for k, v in log_dict.items()}
        self.log_dict(train_log_dict)


    def validation_step(self, batch, batch_index):
        log_dict = self.share_step(batch)
        val_log_dict = {f'val_{k}': v for k, v in log_dict.items()}
        self.log_dict(val_log_dict)


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
            {'params': leres_decoder_params,
             'lr': args.base_lr,
             'weight_decay': args.weight_decay},
            {'params': leres_decoder_params,
             'lr': args.base_lr * args.scale_decoder_lr,
             'weight_decay': args.weight_decay}
        ]

        hmr_generator_leres_opt = torch.optim.SGD(
            hmr_generator_leres_params, momentum=0.9
        )
        hmr_discriminator_opt = torch.optim.Adam(
            self.hmr_discriminator.parameters(),
            lr=args.d_lr,
            weight_decay=args.d_wd
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

# def training_step(self, batch, batch_index):
#     gta_data = batch['gta_loader']
#     mesh_data = batch['mesh_loader']
#     hmr_images = gta_data['hmr_image']
#
#     gt_smpl_theta = gta_data['theta']
#     gt_smpl_shapes = gt_smpl_theta[:, 75:].contiguous()
#     gt_smpl_poses = gt_smpl_theta[:, 3:75].contiguous()
#     gt_smpl_transl = gt_smpl_theta[:, :3].contiguous()
#     gt_kpts_2d = gta_data['joints_2d_origin']
#     gt_kpts_3d = gta_data['joints_3d']
#     gt_intrinsic = gta_data['intrinsic']
#     gt_focal_length = gta_data['focal_length']
#     top, left, height, width = gta_data['leres_cut_box'][:, 0], gta_data['leres_cut_box'][:, 1], \
#                                gta_data['leres_cut_box'][:, 2], gta_data['leres_cut_box'][:, 3]
#
#     predict_smpl_thetas = self.hmr_generator(hmr_images)[-1]
#     predict_smpl_transl = predict_smpl_thetas[:, :3].contiguous()
#     predict_smpl_poses = predict_smpl_thetas[:, 3:75].contiguous()
#     predict_smpl_shapes = predict_smpl_thetas[:, 75:].contiguous()
#
#     predict_kpts_2d, predict_kpts_3d, predict_verts = self.get_smpl_kpts_verts(transl=gt_smpl_transl,
#                                                                                pose=predict_smpl_poses,
#                                                                                shape=predict_smpl_shapes,
#                                                                                focal_length=gt_focal_length)
#
#     height_ratio = self.gta_dataset.leres_size / height
#     width_ratio = self.gta_dataset.leres_size / width
#     predict_kpts_2d[:, :, 0] -= left[:, None]
#     predict_kpts_2d[:, :, 1] -= top[:, None]
#     predict_kpts_2d[:, :, 0] *= height_ratio[:, None]
#     predict_kpts_2d[:, :, 1] *= width_ratio[:, None]
#
#     loss_shape = self.hmr_loss.shape_loss(gt_smpl_shapes, predict_smpl_shapes) * args.e_shape_weight
#     loss_pose = self.hmr_loss.pose_loss(gt_smpl_poses, predict_smpl_poses) * args.e_pose_weight
#     loss_kpts_2d = self.hmr_loss.batch_kp_2d_l1_loss(gt_kpts_2d, predict_kpts_2d) * args.e_2d_kpts_weight
#     # loss_kpts_2d = 0.
#
#     loss_kpts_3d = self.hmr_loss.batch_kp_3d_l2_loss(gt_kpts_3d, predict_kpts_3d) * args.e_3d_kpts_weight
#
#     predict_smpl_thetas[:, :3] = gt_smpl_transl
#     loss_generator_disc = self.hmr_loss.batch_encoder_disc_l2_loss(
#         self.hmr_discriminator(predict_smpl_thetas))
#
#     real_thetas = mesh_data['theta']
#     fake_thetas = predict_smpl_thetas.detach()
#     fake_disc_value, real_disc_value = self.hmr_discriminator(fake_thetas), self.hmr_discriminator(real_thetas)
#     d_disc_real, d_disc_fake, d_disc_loss = self.hmr_loss.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
#
#     loss_generator = (loss_shape + loss_pose + loss_kpts_2d + loss_kpts_3d) * args.e_loss_weight + \
#                      loss_generator_disc * args.d_loss_weight
#
#     loss_discriminator = d_disc_loss * args.d_loss_weight
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
#     leres_log_dict = {
#         'loss_depth_regression': loss_depth_regression,
#         'loss_edge_ranking': loss_edge_ranking,
#         'loss_msg': loss_msg,
#         'loss_pwn_edge': loss_pwn_edge,
#         'loss_leres': loss_leres
#     }
#
#     # loss_align
#     loss_align = self.align_loss.batch_align_loss(predict_verts,
#                                                   torch.tensor([self.smpl_model.faces], device=self.device),
#                                                   predict_depth, gta_data)
#     loss_inside = 0.
#     loss_combie = loss_align + loss_inside
#     combine_log_dict = {
#         'loss_align': loss_align,
#         'loss_inside': loss_inside
#     }
#
#     hmr_generator_leres_opt, hmr_discriminator_opt = self.optimizers()
#
#     hmr_generator_leres_opt.zero_grad()
#     self.manual_backward(loss_generator + loss_align + loss_combie)
#     torch.nn.utils.clip_grad_norm_(self.hmr_generator.parameters(), max_norm=3.0)
#     torch.nn.utils.clip_grad_norm_(self.leres_model.parameters(), max_norm=3.0)
#     hmr_generator_leres_opt.step()
#
#     hmr_discriminator_opt.zero_grad()
#     self.manual_backward(loss_discriminator)
#     torch.nn.utils.clip_grad_norm_(self.hmr_discriminator.parameters(), max_norm=3.0)
#     hmr_discriminator_opt.step()
#
#     hmr_log_dict = {'loss_generator': loss_generator,
#                     'loss_kpts_2d': loss_kpts_2d,
#                     'loss_kpts_3d': loss_kpts_3d,
#                     'loss_shape': loss_shape,
#                     'loss_pose': loss_pose,
#                     'loss_generator_disc': loss_generator_disc,
#                     'loss_discriminator': loss_discriminator,
#                     'd_disc_real': d_disc_real,
#                     'd_disc_fake': d_disc_fake
#                     }
#
#     all_log_dict = {**leres_log_dict, **hmr_log_dict, **combine_log_dict}
#
#     self.log_dict(all_log_dict)
