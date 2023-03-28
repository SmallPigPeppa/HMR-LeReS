import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from d_loss.utils import batch_rodrigues


class HMRLoss(pl.LightningModule):
    def __init__(self):
        super(HMRLoss, self).__init__()

    def batch_kp_2d_l1_loss(self, real_2d_kp, predict_2d_kp):
        kp_gt = real_2d_kp.view(-1, 3)
        kp_pred = predict_2d_kp.contiguous().view(-1, 2)
        vis = kp_gt[:, 2]
        k = torch.sum(vis) * 2.0
        dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
        return torch.matmul(dif_abs, vis) * 1.0 / k



    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp):
        '''
            purpose:
                calc mse * 0.5

            Inputs:
                real_3d_kp  : N x k x 3
                fake_3d_kp  : N x k x 3
                w_3d        : N x 1
        '''
        vis = real_3d_kp[:, :, 3]
        k = torch.sum(vis) * 3.0 * 2.0
        kp_gt = real_3d_kp[:, :, :3]
        kp_pred = fake_3d_kp
        kp_dif = (kp_gt - kp_pred) ** 2
        valid_kp_dif = kp_dif.sum(2) * vis
        # return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
        return torch.sum(valid_kp_dif.sum(1)) * 1.0 / k



    def shape_loss(self, real_shape, fake_shape):
        '''
            Inputs:
                real_shape  :   N x 10
                fake_shape  :   N x 10
        '''
        shape_dif = F.mse_loss(fake_shape, real_shape)
        return shape_dif



    def pose_loss(self, real_pose, fake_pose):
        '''
            Input:
                real_pose   : N x 72
                fake_pose   : N x 72
        '''
        real_rs = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)
        fake_rs = batch_rodrigues(fake_pose.view(-1, 3)).view(-1, 24, 9)
        dif_rs = F.mse_loss(fake_rs, real_rs)
        return dif_rs



    def batch_encoder_disc_l2_loss(self, disc_value):
        '''
            Inputs:
                disc_value: N x 25
        '''
        k = disc_value.shape[0]
        return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k



    def batch_adv_disc_l2_loss(self, real_disc_value, fake_disc_value):
        '''
            Inputs:
                disc_value: N x 25
        '''
        ka = real_disc_value.shape[0]
        kb = fake_disc_value.shape[0]
        lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
        return la, lb, la + lb
