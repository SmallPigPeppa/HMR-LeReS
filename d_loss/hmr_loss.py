import pytorch_lightning as pl
import torch
from utils import batch_rodrigues
class HMRLoss(pl.LightningModule):
    def __init__(self):
        super(HMRLoss, self).__init__()
        self.w_3d=1.0
        self.w_shape=1.0
        self.w_2d=1.0
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

    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp):
        w_3d=self.w_3d
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

    def batch_shape_l2_loss(self, real_shape, fake_shape):
        w_shape=self.w_shape
        k = torch.sum(w_shape) * 10.0 * 2.0 + 1e-8
        shape_dif = (real_shape - fake_shape) ** 2
        return torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k

    '''
        Input:
            real_pose   : N x 72
            fake_pose   : N x 72
    '''

    def batch_pose_l2_loss(self, real_pose, fake_pose):
        w_pose=self.pose
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
