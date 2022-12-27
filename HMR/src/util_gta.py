import cv2
import numpy as np

def flip_image_with_smpl_2dkp(src_image, kps):
    h, w = src_image.shape[0], src_image.shape[1]
    src_image = cv2.flip(src_image, 1)
    if kps is not None:
        kps[:, 0] = w - 1 - kps[:, 0]
        kp_map = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
        kps[:, :] = kps[kp_map]

    return src_image, kps
def reflect_smpl_3dkp(kps):
    kp_map =  [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    joint_ref = kps[kp_map]
    joint_ref[:,0] = -joint_ref[:,0]
    joint_ref[:,:3] = joint_ref[:,:3] - np.mean(joint_ref[:,:3], axis = 0)
    # return joint_ref - np.mean(joint_ref, axis = 0)
    return joint_ref




import torch
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

def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
    shape = real_3d_kp.shape
    vis = real_3d_kp[:, 3]
    # k = torch.sum(w_3d) * shape[1] * 3.0 * 2.0 + 1e-8
    k = torch.sum(w_3d) * torch.sum(vis) * 3.0 * 2.0 + 1e-8
    # first align it
    real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
    kp_gt = real_3d_kp
    kp_pred = fake_3d_kp
    kp_dif = (kp_gt - kp_pred) ** 2
    return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
    return torch.matmul(torch.matmul(kp_dif.sum(1).sum(1), w_3d),vis) * 1.0 / k