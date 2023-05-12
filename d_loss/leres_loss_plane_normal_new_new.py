import torch
import pytorch_lightning as pl
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.ops import estimate_pointcloud_normals


class NormalLoss(pl.LightningModule):
    def __init__(self, min_threshold=0., max_threshold=10.0, scale_factor=5.0):
        super(NormalLoss, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.scale_factor = scale_factor
        self.cameras = None

    def set_cameras_intrinsics(self, intrinsics):
        scaled_intrinsics = intrinsics.clone()
        scaled_intrinsics[:, (0, 1), (0, 1)] /= self.scale_factor  # Update focal length fx, fy
        scaled_intrinsics[:, (0, 1), 2] /= self.scale_factor  # Update principal point cx, cy
        focal_length = scaled_intrinsics[:, (0, 1), (0, 1)]  # fx, fy
        principal_point = scaled_intrinsics[:, (0, 1), 2]  # cx, cy

        if self.cameras is None:
            # If self.cameras is empty, create a new PerspectiveCameras object
            self.cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                              device=self.device)
        else:
            # Otherwise, just update its intrinsics
            self.cameras.focal_length = focal_length
            self.cameras.principal_point = principal_point

    def depth2pcd(self, depth):
        batchsize, height, width = depth.shape
        x = torch.linspace(0, height - 1, height, device=self.device)
        y = torch.linspace(0, width - 1, width, device=self.device)

        grid_x, grid_y = torch.meshgrid(x, y)

        # 扩展坐标网格以匹配批量大小
        grid_x = grid_x.expand(batchsize, height, width)
        grid_y = grid_y.expand(batchsize, height, width)

        # 将 xy_depth 转换为形状为 (B, H, W, 3) 的张量
        xy_depth = torch.stack([grid_x, grid_y, depth], dim=-1)

        # 转换为形状为 (B, N, 3) 的张量，其中 N 是点的数量（H * W）
        xy_depth_flat = xy_depth.view(batchsize, -1, 3)

        # 使用相机反投影点
        point_clouds = self.cameras.unproject_points(xy_depth_flat, world_coordinates=False)

        # 将点云调整为 (B, H, W, 3) 的形状
        point_clouds = point_clouds.view(batchsize, height, width, 3)

        return point_clouds


    def forward(self, depth_pred, depth_gt, intrinsics, plane_mask):
        valid_mask = (depth_gt > self.min_threshold) & (depth_gt < self.max_threshold)
        self.set_cameras_intrinsics(intrinsics)
        pcd_pred = self.depth2pcd(depth_pred)
        pcd_gt = self.depth2pcd(depth_gt)
        batch_size = depth_pred.shape[0]
        loss = torch.tensor(0.0).to(self.device)

        normal_losses = []
        sample_nums = []

        for i in range(batch_size):
            valid_mask_i = valid_mask[i]
            pcd_pred_i = pcd_pred[i]
            pcd_gt_i = pcd_gt[i]
            plane_mask_i = plane_mask[i]
            plane_label_i = torch.unique(plane_mask_i)
            plane_list_i = [plane_mask_i == m for m in plane_label_i if m != 0]
            if len(plane_list_i) == 0:
                continue
            for plane in plane_list_i:
                valid_plane = plane & valid_mask_i
                pcd_gt_plane = pcd_gt_i[valid_plane].view(1, -1, 3)
                pcd_pred_plane = pcd_pred_i[valid_plane].view(1, -1, 3)
                if pcd_pred_plane.shape[1] <= 50:
                    continue
                # Calculate normals
                # a=pcd_gt_plane.shape
                normals_pred_plane = estimate_pointcloud_normals(pcd_gt_plane, neighborhood_size=50)
                normals_gt_plane = estimate_pointcloud_normals(pcd_pred_plane, neighborhood_size=50)
                cos_diff = 1.0 - torch.abs(torch.sum(normals_pred_plane * normals_gt_plane, dim=2))
                normal_loss = torch.mean(cos_diff)
                normal_losses.append(normal_loss)
                sample_nums.append(pcd_pred_plane.shape[1])

        if normal_losses:
            loss = sum(loss * num for loss, num in zip(normal_losses, sample_nums)) / sum(sample_nums)

        return loss


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from datasets.gta_im_all_10801920_plane import GTADataset

    data_dir = '/Users/lwz/torch_ds/gta-im-test/FPS-5'
    # data_dir = '/share/wenzhuoliu/torch_ds/gta-im/FPS-5-test'
    gta_dataset = GTADataset(data_dir)
    gta_loader = DataLoader(gta_dataset, batch_size=2, shuffle=False)
    normal_loss = NormalLoss()

    for i, batch in enumerate(gta_loader):
        depth_gt = batch['depth']
        intrinsics = batch['intrinsic']
        plane_mask = batch['plane_mask']
        a = normal_loss.forward(depth_gt, depth_gt, intrinsics, plane_mask)
        # if i == 1:
        #     break
