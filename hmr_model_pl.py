from model import HMRNetBase
from Discriminator import Discriminator
from hmr_leres_config import args
import hmr_leres_config as config
import torch
from util import align_by_pelvis, batch_rodrigues
from dataloader.mesh_dataloader import mesh_dataloader
from torch.utils.data import DataLoader
from HMR.src.dataloader.gta_dataloader import gta_dataloader as hmr_dataset
import pytorch_lightning as pl
from SMPL import SMPL
from camera_utils import perspective_projection


class HMR(pl.LightningModule):
    def __init__(self):
        super(HMR, self).__init__()
        self.automatic_optimization = False
        self.hmr_generator = HMRNetBase()
        self.hmr_discriminator = Discriminator()
        self.smpl = SMPL(args.smpl_model, obj_saveable=True)

    def train_dataloader(self):
        pix_format = 'NCHW'
        normalize = True
        flip_prob = 0.
        use_flip = False
        hmr_3d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48-add-transl'
        hmr_2d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48-add-transl'
        hmr_mesh_path = 'C:/Users/90532/Desktop/Datasets/HMR/mosh-add-transl'

        # hmr_3d_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48'
        # hmr_2d_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48'
        # hmr_mosh_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/mosh'
        use_crop = True
        scale_range = [1.1, 2.0]
        min_pts_required = 5
        hmr_3d_dataset = hmr_dataset(hmr_3d_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,
                                     normalize, flip_prob)
        hmr_2d_dataset = hmr_dataset(hmr_2d_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,
                                     normalize, flip_prob)
        hmr_mesh_dataset = mesh_dataloader(hmr_mesh_path, use_flip, flip_prob)
        hmr_3d_loader = DataLoader(
            dataset=hmr_3d_dataset,
            batch_size=config.args.batch_3d_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=config.args.num_worker
        )
        hmr_2d_loader = DataLoader(
            dataset=hmr_2d_dataset,
            batch_size=config.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=config.args.num_worker
        )
        hmr_mesh_loader = DataLoader(
            dataset=hmr_mesh_dataset,
            batch_size=config.args.adv_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        loaders = {'hmr_3d_loader': hmr_3d_loader, 'hmr_2d_loader': hmr_2d_loader, 'hmr_mesh_loader': hmr_mesh_loader}
        return loaders

    # def training_step(self, batch, batch_index, optimizer_idx):

    def get_smpl_kpts(self, transl, pose, shape, focal_length, normalize_kpts_2d=True):
        verts, kpts_3d, Rs = self.smpl(beta=shape, theta=pose, get_skin=True)
        kpts_3d += transl
        batch_size = kpts_3d.shape[0]
        kpts_2d = perspective_projection(
            kpts_3d,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=transl,
            focal_length=focal_length,
            camera_center=torch.zeros(batch_size, 2, device=self.device)
        )
        if normalize_kpts_2d:
            kpts_2d = kpts_2d / (self.crop_size / 2.)
        kpts_3d += transl
        verts += transl
        return kpts_2d, kpts_3d

    def training_step(self, batch, batch_index):
        hmr_data = batch['hmr_3d_loader']
        hmr_data_mesh = batch['hmr_mesh_loader']
        images = hmr_data['image']

        gt_smpl_shapes = None
        gt_smpl_poses = None
        w_smpl = None
        w_3d = None
        gt_kpts_2d = None
        gt_kpts_3d = None
        gt_focal_length = None
        predict_smpl_transl = None
        predict_smpl_thetas = self.hmr_generator(images)
        predict_smpl_poses = predict_smpl_thetas[:, 3:75].contiguous()
        predict_smpl_shapes = predict_smpl_thetas[:, 75:].contiguous()
        predict_kpts_2d, predict_kpts_3d = self.get_smpl_kpts(transl=predict_smpl_transl, pose=predict_smpl_poses,
                                                              shape=predict_smpl_shapes, focal_length=gt_focal_length)

        loss_shape = self.batch_shape_l2_loss(gt_smpl_shapes, predict_smpl_shapes,
                                              w_smpl) * args.e_3d_loss_weight * args.e_shape_ratio
        loss_pose = self.batch_pose_l2_loss(gt_smpl_poses, predict_smpl_poses,
                                            w_smpl) * args.e_3d_loss_weight * args.e_pose_ratio
        loss_kpts_2d = self.batch_kp_2d_l1_loss(gt_kpts_2d, predict_kpts_2d) * args.e_loss_weight
        loss_kpts_3d = self.batch_kp_3d_l2_loss(gt_kpts_3d, predict_kpts_3d,
                                              w_3d) * args.e_3d_loss_weight * args.e_3d_kp_ratio

        predict_smpl_thetas[:, 3] = predict_smpl_transl
        loss_generator_disc = self.batch_encoder_disc_l2_loss(
            self.hmr_discriminator(predict_smpl_thetas)) * args.d_loss_weight * args.d_disc_ratio

        real_thetas = hmr_data_mesh['theta']
        fake_thetas = predict_smpl_thetas.detach()
        fake_disc_value, real_disc_value = self.hmr_discriminator(fake_thetas), self.hmr_discriminator(real_thetas)
        d_disc_real, d_disc_fake, d_disc_loss = self.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
        d_disc_real, d_disc_fake, d_disc_loss = d_disc_real * args.d_loss_weight * args.d_disc_ratio, d_disc_fake * args.d_loss_weight * args.d_disc_ratio, d_disc_loss * args.d_loss_w

        loss_generator = loss_shape + loss_pose + loss_kpts_2d + loss_kpts_3d + loss_generator_disc
        loss_discriminator=d_disc_loss



        hmr_generator_opt, hmr_discriminator_opt = self.optimizers()
        hmr_generator_opt.zero_grad()
        self.manual_backward(loss_generator)
        hmr_generator_opt.step()
        hmr_discriminator_opt.zero_grad()
        self.manual_backward(loss_discriminator)
        hmr_discriminator_opt.step()

        hmr_generator_sche, hmr_discriminator_sche = self.lr_schedulers()
        hmr_generator_sche.step()
        hmr_discriminator_sche.step()

        # loss_kp_2d = float(loss_kp_2d)
        # loss_shape = float(loss_shape / args.e_shape_ratio)
        # loss_kp_3d = float(loss_kp_3d / args.e_3d_kp_ratio)
        # loss_pose = float(loss_pose / args.e_pose_ratio)
        # e_disc_loss = float(e_disc_loss / args.d_disc_ratio)
        # d_disc_loss = float(d_disc_loss / args.d_disc_ratio)
        #
        # d_disc_real = float(d_disc_real / args.d_disc_ratio)
        # d_disc_predict = float(d_disc_predict / args.d_disc_ratio)
        #
        # e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss

        iter_msg = {'loss_generator': loss_generator,
                    'loss_kpts_2d': loss_kpts_2d,
                    'loss_kpts_3d': loss_kpts_3d,
                    'loss_kpts_3d': loss_shape,
                    'loss_pose': loss_pose,
                    'loss_generator_disc': loss_generator_disc,
                    'loss_discriminator': loss_discriminator,}
        self.log_dict(iter_msg)

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
        return [hmr_generator_opt, hmr_discriminator_opt], [hmr_generator_sche, hmr_discriminator_sche]

    def _calc_loss(self, generator_outputs, data_2d, data_3d, data_mosh):
        def _accumulate_thetas(generator_outputs):
            thetas = []
            for (theta, verts, j2d, j3d, Rs) in generator_outputs:
                thetas.append(theta)
            return torch.cat(thetas, 0)

        sample_2d_count, sample_3d_count, sample_mosh_count = data_2d['kp_2d'].shape[0], data_3d['kp_2d'].shape[0], \
                                                              data_mosh['theta'].shape
        data_3d_theta, w_3d, w_smpl = data_3d['theta'], data_3d['w_3d'].float(), data_3d[
            'w_smpl'].float()

        total_predict_thetas = _accumulate_thetas(generator_outputs)
        (predict_theta, predict_verts, predict_j2d, predict_j3d, predict_Rs) = generator_outputs[-1]

        real_2d, real_3d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0), data_3d['kp_3d'].float()
        predict_j2d, predict_j3d, predict_theta = predict_j2d, predict_j3d[sample_2d_count:, :], predict_theta[
                                                                                                 sample_2d_count:, :]
        a = predict_j2d * 224 + 122

        # loss_kp_2d = self.batch_kp_2d_l1_loss(real_2d, predict_j2d[:, :14, :]) * args.e_loss_weight
        # loss_kp_3d = self.batch_kp_3d_l2_loss(real_3d, predict_j3d[:, :14, :],
        #                                       w_3d) * args.e_3d_loss_weight * args.e_3d_kp_ratio
        loss_kp_2d = self.batch_kp_2d_l1_loss(real_2d, predict_j2d) * args.e_loss_weight
        loss_kp_3d = self.batch_kp_3d_l2_loss(real_3d, predict_j3d,
                                              w_3d) * args.e_3d_loss_weight * args.e_3d_kp_ratio

        real_shape, predict_shape = data_3d_theta[:, 75:], predict_theta[:, 75:]
        loss_shape = self.batch_shape_l2_loss(real_shape, predict_shape,
                                              w_smpl) * args.e_3d_loss_weight * args.e_shape_ratio

        real_pose, predict_pose = data_3d_theta[:, 3:75], predict_theta[:, 3:75]
        loss_pose = self.batch_pose_l2_loss(real_pose.contiguous(), predict_pose.contiguous(),
                                            w_smpl) * args.e_3d_loss_weight * args.e_pose_ratio

        e_disc_loss = self.batch_encoder_disc_l2_loss(
            self.hmr_discriminator(total_predict_thetas)) * args.d_loss_weight * args.d_disc_ratio

        mosh_real_thetas = data_mosh['theta']
        fake_thetas = total_predict_thetas.detach()
        fake_disc_value, real_disc_value = self.hmr_discriminator(fake_thetas), self.hmr_discriminator(mosh_real_thetas)
        d_disc_real, d_disc_fake, d_disc_loss = self.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
        d_disc_real, d_disc_fake, d_disc_loss = d_disc_real * args.d_loss_weight * args.d_disc_ratio, d_disc_fake * args.d_loss_weight * args.d_disc_ratio, d_disc_loss * args.d_loss_weight * args.d_disc_ratio

        return loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss, d_disc_real, d_disc_fake

    """
        purpose:
            calc L1 error
        Inputs:
            kp_gt  : N x K x 3
            kp_pred: N x K x 2
    """

    def batch_kp_2d_l1_loss(self, real_2d_kp, predict_2d_kp):
        kp_gt = real_2d_kp.view(-1, 3)
        kp_pred = predict_2d_kp.contiguous().view(-1, 2)
        vis = kp_gt[:, 2]
        k = torch.sum(vis) * 2.0 + 1e-8
        dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
        return torch.matmul(dif_abs, vis) * 1.0 / k

    '''
        purpose:
            calc mse * 0.5

        Inputs:
            real_3d_kp  : N x k x 3
            fake_3d_kp  : N x k x 3
            w_3d        : N x 1
    '''

    # def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
    #     shape = real_3d_kp.shape
    #     k = torch.sum(w_3d) * shape[1] * 3.0 * 2.0 + 1e-8
    #     vis = real_3d_kp[:, 3]
    #     # first align it
    #     real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
    #     kp_gt = real_3d_kp
    #     kp_pred = fake_3d_kp
    #     # kp_dif = (kp_gt - kp_pred) ** 2
    #     kp_dif = (kp_gt[:, :, :3] - kp_pred) ** 2
    #     # return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
    #
    #     return torch.matmul(torch.matmul(kp_dif.sum(1).sum(1), w_3d), vis) * 1.0 / k

    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
        shape = real_3d_kp.shape
        k = torch.sum(w_3d) * shape[1] * 3.0 * 2.0 + 1e-8
        vis = real_3d_kp[:, :, 3]
        # first align it
        # real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
        kp_gt = real_3d_kp[:, :, :3]
        kp_pred = fake_3d_kp
        kp_dif = (kp_gt - kp_pred) ** 2
        valid_kp_dif = kp_dif.sum(2) * vis
        # return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
        return torch.matmul(valid_kp_dif.sum(1), w_3d) * 1.0 / k

    '''
        purpose:
            calc mse * 0.5

        Inputs:
            real_shape  :   N x 10
            fake_shape  :   N x 10
            w_shape     :   N x 1
    '''

    def batch_shape_l2_loss(self, real_shape, fake_shape, w_shape):
        k = torch.sum(w_shape) * 10.0 * 2.0 + 1e-8
        shape_dif = (real_shape - fake_shape) ** 2
        return torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k

    '''
        Input:
            real_pose   : N x 72
            fake_pose   : N x 72
    '''

    def batch_pose_l2_loss(self, real_pose, fake_pose, w_pose):
        k = torch.sum(w_pose) * 207.0 * 2.0 + 1e-8
        real_rs, fake_rs = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)[:, 1:, :], batch_rodrigues(
            fake_pose.view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        dif_rs = ((real_rs - fake_rs) ** 2).view(-1, 207)
        return torch.matmul(dif_rs.sum(1), w_pose) * 1.0 / k

    '''
        Inputs:
            disc_value: N x 25
    '''

    def batch_encoder_disc_l2_loss(self, disc_value):
        k = disc_value.shape[0]
        return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k

    '''
        Inputs:
            disc_value: N x 25
    '''

    def batch_adv_disc_l2_loss(self, real_disc_value, fake_disc_value):
        ka = real_disc_value.shape[0]
        kb = fake_disc_value.shape[0]
        lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
        return la, lb, la + lb
