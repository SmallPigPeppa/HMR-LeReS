from scipy.ndimage.morphology import distance_transform_edt
import torch
import torch.nn.functional as F
import FastGeodis


def silhouette_loss(alpha, gt_mask, loss_mask=None, kernel_size=7, edt_power=0.25, l2_weight=0.01, edge_weight=0.01):
    """
    Inputs:
        alpha: Bx1xHxW Tensor, predicted alpha,
        gt_mask: Bx1xHxW Tensor, ground-truth mask
        loss_mask[Optional]: Bx1xHxW Tensor, loss mask, calculate loss inside the mask only
        kernel_size: edge filter kernel size
        edt_power: edge distance power in the loss
        l2_weight: loss weight of the l2 loss
        edge_weight: loss weight of the edge loss
    Output:
        loss
    """
    sil_l2loss = (gt_mask - alpha) ** 2
    if loss_mask is not None:
        sil_l2loss = sil_l2loss * loss_mask

    def compute_edge(x):
        return F.max_pool2d(x, kernel_size, 1, kernel_size // 2) - x

    gt_edge = compute_edge(gt_mask.unsqueeze(1)).cpu().numpy()
    pred_edge = compute_edge(alpha.unsqueeze(1))
    edt = torch.tensor(distance_transform_edt(1 - (gt_edge > 0)) ** (edt_power * 2), dtype=torch.float32,
                       device=pred_edge.device)
    if loss_mask is not None:
        pred_edge = pred_edge * loss_mask.unsqueeze(1)
    sil_edgeloss = torch.sum(pred_edge * edt) / (pred_edge.sum() + 1e-7)
    return sil_l2loss.mean() * l2_weight + sil_edgeloss * edge_weight


import pytorch_lightning as pl


class AlignLoss(pl.LightningModule):
    def __init__(self):
        super(AlignLoss, self).__init__()
        self.kernal_size = 7
        self.edt_power = 0.25
        self.iteration = 2
        self.v = 1e10
        self.lamb = 0.0  # lamb = 0.0 (Euclidean) or 1.0 (Geodesic) or (0.0, 1.0) (mixture)

    def mesh_2d_align_loss(self, gt_mask, pred_mask, valid_mask=None):
        gt_mask_edge = F.max_pool2d(gt_mask, self.kernel_size, 1, self.kernel_size // 2) - gt_mask
        pred_mask_edge = F.max_pool2d(pred_mask, self.kernel_size, 1, self.kernel_size // 2) - pred_mask
        gt_euclidean_dist = FastGeodis.generalised_geodesic2d(1 - (gt_mask_edge > 0), gt_mask_edge, self.v, self.lamb,
                                                              self.iterations)
        gt_euclidean_dist = gt_euclidean_dist ** 2
        edge_align_loss = torch.mean(pred_mask_edge * gt_euclidean_dist)
        mask_align_loss = F.mse_loss(gt_mask, pred_mask)
