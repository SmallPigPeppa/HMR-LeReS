import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.ops import estimate_pointcloud_normals


class DepthRegressionLoss(pl.LightningModule):

    def __init__(self, min_threshold=0., max_threshold=10.0):
        super(DepthRegressionLoss, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def forward(self, pred_depth, gt_depth):
        valid_mask = (gt_depth > self.min_threshold) & (gt_depth < self.max_threshold)  # [b, c, h, w]
        if not torch.any(valid_mask):
            return 0.0
        valid_pred_depth = pred_depth[valid_mask]
        valid_gt_depth = gt_depth[valid_mask]
        l1_loss = F.l1_loss(valid_pred_depth, valid_gt_depth)
        tanh_valid_pred_depth = torch.tanh(0.01 * valid_pred_depth)
        tanh_valid_gt_depth = torch.tanh(0.01 * valid_gt_depth)
        tanh_l1_loss = F.l1_loss(tanh_valid_pred_depth, tanh_valid_gt_depth)
        return l1_loss + tanh_l1_loss
        # return l1_loss


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

    def compute_normals(self, pcd):
        batchsize = pcd.shape[0]
        point_cloud_flat = pcd.view(batchsize, -1, 3)
        # Calculate normals
        normals_flat = estimate_pointcloud_normals(point_cloud_flat, neighborhood_size=50)
        # normals_flat = estimate_pointcloud_normals(point_cloud_flat, neighborhood_size=5)

        # Convert normals back to original depth image shape
        normals = normals_flat.view(pcd.shape)

        return normals

    def forward(self, depth_pred, depth_gt, intrinsics, plane_mask):
        valid_mask = (depth_gt > self.min_threshold) & (depth_gt < self.max_threshold) & (plane_mask > 0)
        if not torch.any(valid_mask):
            return 0.0
        self.set_cameras_intrinsics(intrinsics)
        pcd_pred = self.depth2pcd(depth_pred)
        pcd_gt = self.depth2pcd(depth_gt)
        normals_pred = self.compute_normals(pcd_pred)
        normals_gt = self.compute_normals(pcd_gt)
        cos_diff = 1.0 - torch.abs(torch.sum(normals_pred * normals_gt, dim=3))
        normal_loss = torch.mean(cos_diff[valid_mask])
        return normal_loss


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
        a = normal_loss.forward(depth_gt, depth_gt, intrinsics)
        if i == 1:
            break
