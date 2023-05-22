from datasets.gta_im_all_10801920_plane import GTADataset
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from kornia.contrib import distance_transform


class AlignLoss2(pl.LightningModule):
    def __init__(self):
        super(AlignLoss2, self).__init__()
        self.kernal_size = 7
        self.edt_power = 0.5

    def distance_transform_manhattan(self, image: torch.Tensor) -> torch.Tensor:
        out = distance_transform(image, kernel_size=3, h=0.35)
        return out

    def distance_transform_euclidean(self, image: torch.Tensor) -> torch.Tensor:
        euclidean_dist = distance_transform_edt(1 - image.cpu().numpy())
        out = torch.tensor(euclidean_dist, dtype=torch.float32, device=self.device)
        return out

    def mesh_2d_align_loss_new(self, gt_mask, pred_mask, valid_mask=None):
        gt_mask_edge = F.max_pool2d(gt_mask, self.kernal_size, 1, self.kernal_size // 2) - gt_mask
        pred_mask_edge = F.max_pool2d(pred_mask, self.kernal_size, 1, self.kernal_size // 2) - pred_mask

        gt_edge_dist = self.distance_transform_manhattan(gt_mask_edge)
        # gt_edge_dist = self.distance_transform_euclidean(gt_mask_edge)
        gt_edge_dist = gt_edge_dist ** self.edt_power

        mask_edge_loss = torch.mean(pred_mask_edge * gt_euclidean_dist)
        mask_l2_loss = F.mse_loss(gt_mask, pred_mask)
        return mask_edge_loss, mask_l2_loss

    def mesh_2d_align_loss(self, gt_mask, pred_mask, valid_mask=None):
        gt_mask_edge = F.max_pool2d(gt_mask, self.kernal_size, 1, self.kernal_size // 2) - gt_mask
        pred_mask_edge = F.max_pool2d(pred_mask, self.kernal_size, 1, self.kernal_size // 2) - pred_mask
        gt_euclidean_dist = self.distance_transform_euclidean(gt_mask_edge)**0.5
        gt_manhattan_dist = self.distance_transform_manhattan(gt_mask_edge)**0.5
        # gt_euclidean_dist = distance_transform_edt(1 - gt_mask.cpu().numpy())
        a=pred_mask_edge * gt_euclidean_dist
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
        human_mask = human_mask.unsqueeze(1)
        gt_euclidean_dist, gt_manhattan_dist = model_c.mesh_2d_align_loss(gt_mask=human_mask.to(float),
                                                                          pred_mask=human_mask.to(float))

        a = gt_euclidean_dist.cpu().numpy()
        b = gt_manhattan_dist.cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # ax[0].imshow(leres_image[0, 0], cmap='gray')
        # ax[0].set_title('Input Image')
        ax[0].imshow(human_mask[0, 0], cmap='gray')
        ax[0].set_title('Input Image')

        ax[1].imshow(gt_euclidean_dist[0, 0], cmap='jet')
        ax[1].set_title('Euclidean Distance Transform')

        ax[2].imshow(gt_manhattan_dist[0, 0], cmap='jet')
        ax[2].set_title('Manhattan Distance Transform')
        plt.show()
        if idx > 4:
            break
