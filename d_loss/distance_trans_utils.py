import math

import torch

from kornia.filters import filter2d
from kornia.utils import create_meshgrid
from scipy.ndimage.morphology import distance_transform_edt

from d_loss.align_loss_new_debug import AlignLoss
from datasets.gta_im_all_10801920_plane import GTADataset
from torch.utils.data import DataLoader
from hmr_leres_model_all_fast_debug_plane_origin import HMRLeReS
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from kornia.contrib import distance_transform


class AlignLoss2(pl.LightningModule):
    def __init__(self):
        super(AlignLoss2, self).__init__()
        self.kernel_size = 7
        self.edt_power = 0.5

    def distance_transform_manhattan(self, image: torch.Tensor) -> torch.Tensor:
        out = distance_transform(image, kernel_size=3, h=0.35)
        return out

    def distance_transform_euclidean(self, image: torch.Tensor) -> torch.Tensor:
        euclidean_dist = distance_transform_edt(1 - image.cpu().numpy())
        out = torch.tensor(euclidean_dist, dtype=torch.float32, device=self.device)
        return out

    def mesh_2d_align_loss(self, gt_mask, pred_mask, valid_mask=None):
        gt_mask_edge = F.max_pool2d(gt_mask, self.kernel_size, 1, self.kernel_size // 2) - gt_mask
        pred_mask_edge = F.max_pool2d(pred_mask, self.kernel_size, 1, self.kernel_size // 2) - pred_mask
        gt_euclidean_dist = self.distance_transform_euclidean(gt_mask_edge)
        # gt_euclidean_dist = gt_euclidean_dist ** 2
        gt_manhattan_dist = self.distance_transform_manhattan(gt_mask_edge)
        gt_manhattan_dist = gt_manhattan_dist ** self.edt_power

        edge_align_loss = torch.mean(pred_mask_edge * gt_euclidean_dist)
        mask_align_loss = F.mse_loss(gt_mask, pred_mask)
        return gt_euclidean_dist, gt_manhattan_dist


if __name__ == "__main__":
    data_dir = '/Users/lwz/torch_ds/gta-im-test/FPS-5'
    gta_dataset = GTADataset(data_dir)
    gta_loader = DataLoader(gta_dataset, batch_size=4, shuffle=False)
    model_c = AlignLoss2()
    for idx, batch in enumerate(gta_loader):
        depth = batch['depth']
        leres_image = batch['leres_image']
        human_mask = batch['human_mask']
        gt_euclidean_dist, gt_manhattan_dist = model_c.mesh_2d_align_loss(gt_mask=human_mask, pred_mask=human_mask)
        if idx > 4:
            break
