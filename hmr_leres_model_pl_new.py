import torch.nn.functional
from hmr_leres_config import args
import hmr_leres_config as config
from lib_train.models import network_auxi as network
from lib_train.utils.net_tools import *
from lib_train.models.PWN_edges import EdgeguidedNormalRegressionLoss
from lib_train.models.ranking_loss import EdgeguidedRankingLoss
from lib_train.models.ILNR_loss import MEADSTD_TANH_NORM_Loss, DepthRegressionLoss
from lib_train.models.MSGIL_loss import MSGIL_NORM_Loss
import pytorch_lightning as pl
from dataloader.mesh_dataloader import mesh_dataloader
from torch.utils.data import DataLoader
from HMR.src.dataloader.gta_dataloader import gta_dataloader as hmr_dataset
from LeReS.Train.data.gta_dataset import GTADataset as leres_gta_dataset
from pytorch_lightning.trainer.supporters import CombinedLoader
from leres_val_utils import validate_rel_depth_err
from util import align_by_pelvis, batch_rodrigues
from model import HMRNetBase
from Discriminator import Discriminator


class HMRLeReS(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # leres
        self.depth_model = DepthModel()
        # leres pair-wise normal d_loss (pwn edge)
        self.pn_edge = EdgeguidedNormalRegressionLoss(mask_value=-1e-8, max_threshold=15)
        # leres multi-scale gradient d_loss (msg)
        self.msg_normal_loss = MSGIL_NORM_Loss(scale=4, valid_threshold=-1e-8)
        # Scale shift invariant. SSIMAEL_Loss is MIDAS d_loss. MEADSTD_TANH_NORM_Loss is our normalization d_loss.
        self.meanstd_tanh_loss = MEADSTD_TANH_NORM_Loss(valid_threshold=-1e-8)
        self.depth_regression_loss = DepthRegressionLoss(valid_threshold=-1e-8, max_threshold=15)
        self.ranking_edge_loss = EdgeguidedRankingLoss(mask_value=-1e-8)

        # hmr
        self.hmr_generator = HMRNetBase()
        self.hmr_discriminator = Discriminator()
        self.automatic_optimization = False

    def train_dataloader(self):
        hmr_3d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
        hmr_2d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
        hmr_mosh_path = 'C:/Users/90532/Desktop/Datasets/HMR/mosh'

        # hmr_3d_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48'
        # hmr_2d_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48'
        # hmr_mosh_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/mosh'
        pix_format = 'NCHW'
        normalize = True
        flip_prob = 0.
        use_flip = False
        use_crop = True
        scale_range = [1.1, 2.0]
        min_pts_required = 5
        hmr_3d_dataset = hmr_dataset(hmr_3d_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,
                                     normalize, flip_prob)
        hmr_2d_dataset = hmr_dataset(hmr_2d_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,
                                     normalize, flip_prob)
        hmr_mosh_dataset = mesh_dataloader(hmr_mosh_path, use_flip, flip_prob)
        leres_dataset = leres_gta_dataset(config.args, '2020-06-11-10-06-48')

        hmr_3d_loader = DataLoader(
            dataset=hmr_3d_dataset,
            batch_size=config.args.batch_3d_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=config.args.num_worker
        )
        hmr_2d_loader = DataLoader(
            dataset=hmr_2d_dataset,
            batch_size=config.args.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=config.args.num_worker
        )
        hmr_mosh_loader = DataLoader(
            dataset=hmr_mosh_dataset,
            batch_size=config.args.adv_batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        leres_loader = DataLoader(
            dataset=leres_dataset,
            batch_size=config.args.batchsize,
            num_workers=config.args.num_worker,
            shuffle=False,
            drop_last=True,
            pin_memory=True, )
        loaders = {'hmr_3d_loader': hmr_3d_loader, 'hmr_2d_loader': hmr_2d_loader, 'hmr_mosh_loader': hmr_mosh_loader,
                   'leres_loader': leres_loader}
        return loaders

    def val_dataloader(self):
        hmr_3d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
        hmr_2d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
        hmr_mosh_path = 'C:/Users/90532/Desktop/Datasets/HMR/mosh'

        # hmr_3d_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48'
        # hmr_2d_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48'
        # hmr_mosh_path = '/share/wenzhuoliu/torch_ds/HMR-LeReS/mosh'
        pix_format = 'NCHW'
        normalize = True
        flip_prob = 0.
        use_flip = False
        use_crop = True
        scale_range = [1.1, 2.0]
        min_pts_required = 5
        hmr_3d_dataset = hmr_dataset(hmr_3d_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,
                                     normalize, flip_prob)
        hmr_2d_dataset = hmr_dataset(hmr_2d_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,
                                     normalize, flip_prob)
        hmr_mosh_dataset = mesh_dataloader(hmr_mosh_path, use_flip, flip_prob)
        leres_dataset = leres_gta_dataset(config.args, '2020-06-11-10-06-48')

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
        hmr_mosh_loader = DataLoader(
            dataset=hmr_mosh_dataset,
            batch_size=config.args.adv_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        leres_loader = DataLoader(
            dataset=leres_dataset,
            batch_size=config.args.batchsize,
            num_workers=config.args.num_worker,
            shuffle=True,
            drop_last=True,
            pin_memory=True, )
        loaders = {'hmr_3d_loader': hmr_3d_loader, 'hmr_2d_loader': hmr_2d_loader, 'hmr_mosh_loader': hmr_mosh_loader,
                   'leres_loader': leres_loader}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def leres_forward(self, data):
        inputs = data['rgb']
        logit, auxi = self.depth_model(inputs)
        losses_dict = self.leres_loss(logit, data)
        return {'decoder': logit, 'auxi': auxi, 'losses': losses_dict}

    def leres_inference(self, data):
        with torch.no_grad():
            out = self.forward(data)
            pred_depth = out['decoder']
            pred_disp = out['auxi']
            pred_depth_out = pred_depth
            return {'pred_depth': pred_depth_out, 'pred_disp': pred_disp}

    def leres_loss(self, pred_logit, data):
        pred_depth = pred_logit
        gt_depth = data['depth']
        gt_depth_mid = gt_depth
        pred_depth_mid = pred_depth

        if gt_depth_mid.ndim == 3:
            gt_depth_mid = gt_depth_mid[None, :, :, :]
            pred_depth_mid = pred_depth_mid[None, :, :, :]

        loss = {}
        # d_loss['meanstd-tanh_loss'] = self.meanstd_tanh_loss(pred_depth_mid, gt_depth_mid)  # L-ILNR
        loss['depth-regression-d_loss'] = self.depth_regression_loss(pred_depth_mid, gt_depth_mid)  # L-ILNR

        # d_loss['ranking-edge_loss'] = self.ranking_edge_loss(pred_depth, gt_depth, data['rgb'])
        #
        # d_loss['msg_normal_loss'] = (self.msg_normal_loss(pred_depth_mid, gt_depth_mid) * 0.5)
        #
        # pred_ssinv = recover_scale_shift_depth(pred_depth, gt_depth, min_threshold=-1e-8, max_threshold=10.1)
        # d_loss['pairwise-normal-regress-edge_loss'] = self.pn_edge(pred_ssinv,
        #                                                          gt_depth,
        #                                                          data['rgb'],
        #                                                          focal_length=data['focal_length'])
        total_loss = sum(loss.values())
        loss['total_loss'] = total_loss
        return loss

    def training_step(self, batch, batch_index):

        # hmr d_loss
        hmr_data_3d = batch['hmr_3d_loader']
        hmr_data_2d = batch['hmr_2d_loader']
        hmr_data_mosh = batch['hmr_mosh_loader']
        hmr_image_from_2d, hmr_image_from_3d = hmr_data_2d['image'], hmr_data_3d['image']
        hmr_images = torch.cat((hmr_image_from_2d, hmr_image_from_3d), dim=0)
        hmr_generator_outputs = self.hmr_generator(hmr_images)
        loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self.calc_loss(
            hmr_generator_outputs, hmr_data_2d, hmr_data_3d, hmr_data_mosh)
        e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss
        d_loss = d_disc_loss

        # leres d_loss
        leres_data = batch['leres_loader']
        leres_out = self.leres_forward(leres_data)
        leres_loss_dict = leres_out['losses']
        leres_loss = leres_loss_dict['total_loss']
        # transfer depth_gt and depth_pred to hmr camera system
        leres_images = leres_data['rgb']
        depth_pred, auxi = self.depth_model(leres_images)
        losses_dict = self.leres_loss(depth_pred, data)
        return {'decoder': logit, 'auxi': auxi, 'losses': losses_dict}


        # hmr-leres d_loss


        hmr_generator_opt, hmr_discriminator_opt, leres_opt = self.optimizers()
        hmr_generator_opt.zero_grad()
        self.manual_backward(e_loss)
        hmr_generator_opt.step()
        hmr_discriminator_opt.zero_grad()
        self.manual_backward(d_loss)
        hmr_discriminator_opt.step()
        leres_opt.zero_grad()
        self.manual_backward(leres_loss)
        leres_opt.step()


        hmr_generator_sche, hmr_discriminator_sche = self.lr_schedulers()
        hmr_generator_sche.step()
        hmr_discriminator_sche.step()

        loss_kp_2d = float(loss_kp_2d)
        loss_shape = float(loss_shape / args.e_shape_ratio)
        loss_kp_3d = float(loss_kp_3d / args.e_3d_kp_ratio)
        loss_pose = float(loss_pose / args.e_pose_ratio)
        e_disc_loss = float(e_disc_loss / args.d_disc_ratio)
        d_disc_loss = float(d_disc_loss / args.d_disc_ratio)

        d_disc_real = float(d_disc_real / args.d_disc_ratio)
        d_disc_predict = float(d_disc_predict / args.d_disc_ratio)

        e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss

        hmr_iter_msg = {'e_loss': e_loss,
                        '2d_loss': loss_kp_2d,
                        '3d_loss': loss_kp_3d,
                        'shape_loss': loss_shape,
                        'pose_loss': loss_pose,
                        'e_disc_loss': float(e_disc_loss),
                        'd_disc_loss': float(d_disc_loss),
                        'd_disc_real': float(d_disc_real),
                        'd_disc_predict': float(d_disc_predict)}
        self.log_dict(hmr_iter_msg)
        self.log_dict(leres_loss_dict)

    def validation_step(self, batch, batch_index):
        criteria_dict = {}
        leres_data = batch['leres_loader']
        pred_depth, _ = self.depth_model(leres_data['rgb'])
        gt_depth = leres_data['depth']
        criteria_dict.update(validate_rel_depth_err(pred=pred_depth, gt=gt_depth))
        self.log_dict(criteria_dict, on_step=True)

        return criteria_dict

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
            if 'depth_model' in key and value.requires_grad:
                if 'res' in key:
                    leres_encoder_params.append(value)
                    leres_encoder_params_names.append(key)
                else:
                    leres_decoder_params.append(value)
                    leres_decoder_params_names.append(key)
            else:
                leres_nograd_param_names.append(key)

        leres_lr_encoder = cfg.TRAIN.BASE_LR
        leres_lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR
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
    def calc_loss(self, generator_outputs, data_2d, data_3d, data_mosh):
        def accumulate_thetas(generator_outputs):
            thetas = []
            for (theta, verts, j2d, j3d, Rs) in generator_outputs:
                thetas.append(theta)
            return torch.cat(thetas, 0)

        sample_2d_count, sample_3d_count, sample_mosh_count = data_2d['kp_2d'].shape[0], data_3d['kp_2d'].shape[0], \
                                                              data_mosh['theta'].shape
        data_3d_theta, w_3d, w_smpl = data_3d['theta'], data_3d['w_3d'].float(), data_3d[
            'w_smpl'].float()

        total_predict_thetas = accumulate_thetas(generator_outputs)
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
        real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
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





class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()
        self.auxi_modules = network.AuxiNetV2()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit, auxi_input = self.decoder_modules(lateral_out)
        out_auxi = self.auxi_modules(auxi_input)
        return out_logit, out_auxi


def recover_scale_shift_depth(pred, gt, min_threshold=1e-8, max_threshold=1e8):
    b, c, h, w = pred.shape
    mask = (gt > min_threshold) & (gt < max_threshold)  # [b, c, h, w]
    EPS = 1e-6 * torch.eye(2, dtype=pred.dtype, device=pred.device)
    scale_shift_batch = []
    ones_img = torch.ones((1, h, w), dtype=pred.dtype, device=pred.device)
    for i in range(b):
        mask_i = mask[i, ...]
        pred_valid_i = pred[i, ...][mask_i]
        ones_i = ones_img[mask_i]
        pred_valid_ones_i = torch.stack((pred_valid_i, ones_i), dim=0)  # [c+1, n]
        A_i = torch.matmul(pred_valid_ones_i, pred_valid_ones_i.permute(1, 0))  # [2, 2]
        A_inverse = torch.inverse(A_i + EPS)

        gt_i = gt[i, ...][mask_i]
        B_i = torch.matmul(pred_valid_ones_i, gt_i)[:, None]  # [2, 1]
        scale_shift_i = torch.matmul(A_inverse, B_i)  # [2, 1]
        scale_shift_batch.append(scale_shift_i)
    scale_shift_batch = torch.stack(scale_shift_batch, dim=0)  # [b, 2, 1]
    ones = torch.ones_like(pred)
    pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
    pred_scale_shift = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2),
                                    scale_shift_batch)  # [b, h*w, 1]
    pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))
    return pred_scale_shift

