import torch.nn.functional
from lib_train.models import network_auxi as network
from lib_train.configs.config import cfg
import hmr_leres_config as config
from lib_train.utils.net_tools import *
from lib_train.models.PWN_edges import EdgeguidedNormalRegressionLoss
from lib_train.models.ranking_loss import EdgeguidedRankingLoss
from lib_train.models.ILNR_loss import MEADSTD_TANH_NORM_Loss
from lib_train.models.MSGIL_loss import MSGIL_NORM_Loss
import pytorch_lightning as pl
from dataloader.mosh_dataloader import mosh_dataloader
from torch.utils.data import DataLoader
from HMR.src.dataloader.gta_dataloader import gta_dataloader as hmr_dataset
from LeReS.Train.data.gta_dataset import GTADataset as leres_gta_dataset


class LeReS(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.depth_model = DepthModel()
        # leres pair-wise normal loss (pwn edge)
        self.pn_edge = EdgeguidedNormalRegressionLoss(mask_value=-1e-8, max_threshold=10.1)
        # leres multi-scale gradient loss (msg)
        self.msg_normal_loss = MSGIL_NORM_Loss(scale=4, valid_threshold=-1e-8)
        # Scale shift invariant. SSIMAEL_Loss is MIDAS loss. MEADSTD_TANH_NORM_Loss is our normalization loss.
        self.meanstd_tanh_loss = MEADSTD_TANH_NORM_Loss(valid_threshold=-1e-8)
        self.ranking_edge_loss = EdgeguidedRankingLoss(mask_value=-1e-8)

    def train_dataloader(self):
        pix_format = 'NCHW'
        normalize = True
        flip_prob = 0.
        use_flip = False
        hmr_3d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
        hmr_2d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
        hmr_mosh_path = 'C:/Users/90532/Desktop/Datasets/HMR/mosh'

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
        hmr_mosh_dataset = mosh_dataloader(hmr_mosh_path, use_flip, flip_prob)
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
        return loaders

    def forward(self, data):
        inputs = data['rgb']
        logit, auxi = self.depth_model(inputs)
        losses_dict = self.loss(logit, data)
        return {'decoder': logit, 'auxi': auxi, 'losses': losses_dict}

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data)
            pred_depth = out['decoder']
            pred_disp = out['auxi']
            pred_depth_out = pred_depth
            return {'pred_depth': pred_depth_out, 'pred_disp': pred_disp}

    def loss(self, pred_logit, data):
        pred_depth = pred_logit
        # gt_depth = data['depth'].to(device=self.device)
        # gt_depth = data['depth'].to(device=pred_depth.device)
        gt_depth = data['depth']
        gt_depth_mid = gt_depth
        pred_depth_mid = pred_depth

        if gt_depth_mid.ndim == 3:
            gt_depth_mid = gt_depth_mid[None, :, :, :]
            pred_depth_mid = pred_depth_mid[None, :, :, :]
        loss = {}

        pred_ssinv = recover_scale_shift_depth(pred_depth, gt_depth, min_threshold=-1e-8, max_threshold=10.1)
        loss['pairwise-normal-regress-edge_loss'] = self.pn_edge(pred_ssinv,
                                                                 gt_depth,
                                                                 data['rgb'],
                                                                 focal_length=data['focal_length'])

        loss_ssi = self.meanstd_tanh_loss(pred_depth_mid, gt_depth_mid)  # L-ILNR
        loss['meanstd-tanh_loss'] = loss_ssi

        loss['ranking-edge_loss'] = self.ranking_edge_loss(pred_depth, gt_depth, data['rgb'])

        loss['msg_normal_loss'] = (self.msg_normal_loss(pred_depth_mid, gt_depth_mid) * 0.5).float()
        # loss['msg_normal_loss'] = self.msg_normal_loss(pred_depth_mid, gt_depth_mid) * 0.5

        total_loss = sum(loss.values())
        loss['total_loss'] = total_loss
        return loss

    def training_step(self, batch, batch_index):
        leres_data = batch['leres_loader']
        out = self.forward(leres_data)
        loss_dict = out['losses']
        self.log('total_loss', loss_dict['total_loss'])
        self.log('pairwise-normal-regress-edge_loss', loss_dict['pairwise-normal-regress-edge_loss'])
        self.log('ranking-edge_loss', loss_dict['ranking-edge_loss'])
        self.log('msg_normal_loss', loss_dict['msg_normal_loss'])
        return loss_dict['total_loss']

    def configure_optimizers(self):
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in self.named_parameters():
            if value.requires_grad:
                if 'res' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder = cfg.TRAIN.BASE_LR
        lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR
        weight_decay = 0.0005

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'weight_decay': weight_decay},
        ]
        optimizer = torch.optim.SGD(net_params, momentum=0.9)
        return optimizer


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


class ModelOptimizer(object):
    def __init__(self, model):
        super(ModelOptimizer, self).__init__()
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in model.named_parameters():
            if value.requires_grad:
                if 'res' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder = cfg.TRAIN.BASE_LR
        lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR
        weight_decay = 0.0005

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'weight_decay': weight_decay},
        ]
        self.optimizer = torch.optim.SGD(net_params, momentum=0.9)
        self.model = model

    def optim(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
