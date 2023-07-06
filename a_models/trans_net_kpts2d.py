import torch
from torch import nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import PointsRasterizationSettings, PointsRenderer, PointsRasterizer
import matplotlib.pyplot as plt
import open3d as o3d


class TransNet(LightningModule):
    def __init__(self, pcd_num=5000):
        super(TransNet, self).__init__()
        self.pcd_num = pcd_num
        self.mlp = nn.Sequential(
            nn.Linear(pcd_num * 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # SMPL transl参数有3个元素
        )

    def perspective_projection(self, points, rotation, translation,
                               focal_length, camera_center):
        """
        This function computes the perspective projection of a set of points.
        Input:
            points (bs, N, 3): 3D points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size = points.shape[0]
        K = torch.zeros([batch_size, 3, 3], device=self.device)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij,bkj->bki', rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:, :, -1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

        return projected_points[:, :, :-1]

    def inverse_perspective_projection(self, points2D, depths, rotation, translation,
                                       focal_length, camera_center):
        """
        This function computes the inverse perspective projection of a set of 2D points.
        Input:
            points2D (bs, N, 2): 2D points
            depths (bs, N): depths of the points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size = points2D.shape[0]
        K_inv = torch.zeros([batch_size, 3, 3], device=self.device)
        K_inv[:, 0, 0] = 1. / focal_length
        K_inv[:, 1, 1] = 1. / focal_length
        K_inv[:, 2, 2] = 1.
        K_inv[:, :-1, -1] = -camera_center / focal_length

        # Add third dimension (depths)
        points3D = torch.cat([points2D, depths.unsqueeze(-1)], dim=-1)

        # Undo camera intrinsics
        points3D = torch.einsum('bij,bkj->bki', K_inv, points3D)

        # Undo perspective distortion
        points3D = points3D * points3D[:, :, -1].unsqueeze(-1)

        # Undo transformation
        points3D = points3D - translation.unsqueeze(1)
        points3D = torch.einsum('bij,bkj->bki', rotation.transpose(1, 2), points3D)

        return points3D

    def depth_to_pointcloud(self, depth_map, rotation, translation, focal_length, camera_center):

        """
        This function computes the 3D points from a depth map.
        Input:
            depth_map (bs, H, W): Depth map
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size, height, width = depth_map.shape
        K_inv = torch.zeros([batch_size, 3, 3], device=self.device)
        K_inv[:, 0, 0] = 1. / focal_length
        K_inv[:, 1, 1] = 1. / focal_length
        K_inv[:, 2, 2] = 1.
        K_inv[:, :-1, -1] = -camera_center / focal_length.unsqueeze(-1).expand(-1, 2)

        # Create a grid of 2D points
        x = torch.linspace(0, width - 1, width, device=self.device)
        y = torch.linspace(0, height - 1, height, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        points2D = torch.stack((grid_x, grid_y), dim=-1)  # (H*W, 2)

        # Repeat points for each item in the batch
        points2D = points2D.unsqueeze(0).repeat(batch_size, 1, 1)

        # Concatenate depth map
        depths = depth_map.reshape(batch_size, -1)  # (bs, H*W)
        points3D = torch.cat([points2D, depths.unsqueeze(-1)], dim=-1)  # (bs, H*W, 3)

        # Undo camera intrinsics
        points3D = torch.einsum('bij,bkj->bki', K_inv, points3D)

        # Undo perspective distortion
        # points3D = points3D * points3D[:, :, -1].unsqueeze(-1)
        #
        # # Undo transformation
        # points3D = points3D - translation.unsqueeze(1)
        # points3D = torch.einsum('bij,bkj->bki', rotation.transpose(1, 2), points3D)

        return points3D.reshape(batch_size, height, width, 3)  # (bs, H, W, 3)

    def forward(self, x):
        # x是形状为(batch_size, 5000, 3)的点云数据
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.mlp(x)
        return x

    def random_sample(self, human_pcd, make_noise=False):
        if human_pcd.shape[0] < 5000:
            # 如果点云数量小于5000，进行有放回的采样
            random_indices = torch.randint(human_pcd.shape[0], (5000,))
        else:
            random_indices = torch.randperm(human_pcd.shape[0])[:5000]

        sampled_points = human_pcd[random_indices]

        # 可选：添加一些随机噪声
        if make_noise:
            noise = torch.randn_like(sampled_points) * 0.05
            sampled_points += noise

        return sampled_points

    def get_pcd(self, depth, focal_length):
        batchsize, height, width = depth.shape
        # 通过相机内参矩阵生成相机参数
        cameras = PerspectiveCameras(focal_length=focal_length,
                                     principal_point=(height // 2, width // 2),
                                     image_size=((height, width)),
                                     device=self.device)

        # 通过深度图生成3D点云
        xyz = cameras.unproject_points(depth)

    def get_human_pcd(self, depth, human_mask, focal_length):
        # K是相机内参矩阵
        height, width = depth.shape
        # 通过相机内参矩阵生成相机参数
        cameras = PerspectiveCameras(focal_length=focal_length,
                                     principal_point=(height // 2, width // 2),
                                     image_size=((height, width)),
                                     device=self.device)

        # 通过深度图生成3D点云
        xyz = cameras.unproject_points(depth.unsqueeze(-1))

        # 使用人类掩码提取人的点云
        human_pcd = xyz[human_mask]

        return human_pcd

    def convert_intrinsic(self, intrinsic):
        batchsize = intrinsic.shape[0]
        intrinsic_4x4 = torch.zeros(batchsize, 4, 4, dtype=intrinsic.dtype, device=self.device)
        intrinsic_4x4[:, :3, :3] = intrinsic
        intrinsic_4x4[:, 2, 3] = 1
        intrinsic_4x4[:, 3, 2] = 1
        return intrinsic_4x4

    def get_pcd2(self, depth, intrinsic, human_mask=None):
        batchsize, height, width = depth.shape
        image_size = (height, width)
        # cam_rotate = torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]], dtype=intrinsic.dtype, device=self.device)
        cam_rotate = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=intrinsic.dtype, device=self.device)
        cam_trans = torch.zeros((1, 3), dtype=intrinsic.dtype, device=self.device)
        intrinsic_4x4 = self.convert_intrinsic(intrinsic)
        cameras = PerspectiveCameras(device=self.device, R=cam_rotate, T=cam_trans, K=intrinsic_4x4,
                                     image_size=[image_size], in_ndc=False)

        # 创建像素的网格
        x = torch.linspace(0, width - 1, width).repeat(height, 1).reshape(-1).to(self.device)
        y = torch.linspace(0, height - 1, height).repeat(width, 1).t().reshape(-1).to(self.device)

        # 对所有batch进行操作
        x = x.repeat(batchsize)
        y = y.repeat(batchsize)

        # 重塑深度图
        z = depth.reshape(batchsize * height * width)

        # 构建点云，每个点由x, y, z坐标表示
        xy_depth = torch.stack((x, y, z), dim=1)
        b = xy_depth.reshape(batchsize, height, width, 3)[0].detach().numpy()

        # 重塑为(Batch Size, N, 3)的形式
        xy_depth = xy_depth.reshape(batchsize, -1, 3)
        xyz = cameras.unproject_points(xy_depth)
        xyz_np = xyz[0].cpu().numpy()
        xyz_np = xyz_np.reshape(-1, 3)

        # 创建一个 Open3D PointCloud 对象
        pcd = o3d.geometry.PointCloud()

        # 使用 xyz 数据更新 PointCloud
        pcd.points = o3d.utility.Vector3dVector(xyz_np)

        # 可视化点云
        # o3d.visualization.draw_geometries([pcd])

        projected_xyz = self.perspective_projection(
            xyz,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batchsize, -1, -1),
            translation=torch.zeros(size=[batchsize, 3], device=self.device),
            focal_length=focal_length,
            camera_center=torch.Tensor([384 // 2, 216 // 2]).to(self.device).expand(batchsize, -1)
        )
        a = projected_xyz[0].detach().cpu().numpy().reshape(216, 384, 2)

        projected_xyz_human_0 = projected_xyz[0].reshape(216, 384, 2)[human_mask[0]].long()

        # projected_xyz_human_0xyz_np = xyz[0].cpu().numpy()
        # xyz_np = xyz_np.reshape(-1, 3)

        # 创建一个 Open3D PointCloud 对象
        # pcd = o3d.geometry.PointCloud()
        #
        # # 使用 xyz 数据更新 PointCloud
        # pcd.points = o3d.utility.Vector3dVector(xyz_np)
        #
        # # 可视化点云
        # o3d.visualization.draw_geometries([pcd])

        # 创建一个全零的RGB图像
        rendered_image = torch.zeros(height, width, 3)

        # 将对应的位置设置为红色
        rendered_image[projected_xyz_human_0[:, 1], projected_xyz_human_0[:, 0], 0] = 255

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(depth[0].detach().cpu().numpy(), cmap='gray')
        plt.title("Original depth image")

        plt.subplot(1, 2, 2)
        plt.imshow(rendered_image.detach().cpu().numpy())
        plt.title("Rendered image")
        plt.show()

    def get_pcd_kpts2d(self, depth, intrinsic, kpts2d=None):
        batchsize, height, width = depth.shape
        image_size = (height, width)
        cam_rotate = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=intrinsic.dtype, device=self.device)
        cam_trans = torch.zeros((1, 3), dtype=intrinsic.dtype, device=self.device)
        intrinsic_4x4 = self.convert_intrinsic(intrinsic)
        cameras = PerspectiveCameras(device=self.device, R=cam_rotate, T=cam_trans, K=intrinsic_4x4,
                                     image_size=[image_size], in_ndc=False)

        kpts_x, kpts_y, valid = kpts2d[..., 0].long(), kpts2d[..., 1].long(), kpts2d[..., 2]
        valid *= ((kpts_x >= 0) & (kpts_x < height) & (kpts_y >= 0) & (kpts_y < width)).float()
        # Clamp kpts_x and kpts_y to prevent index out of bounds
        kpts_x = kpts_x.clamp(0, height - 1)
        kpts_y = kpts_y.clamp(0, width - 1)

        # Use advanced indexing to replace the loop
        # valid = valid.float().to(self.device)

        # Use advanced indexing to replace the loop
        z = depth[torch.arange(batchsize).unsqueeze(1), kpts_x, kpts_y]
        xy_depth = torch.stack((kpts_x.float(), kpts_y.float(), z), dim=-1)

        # 使用unproject_points转换为3D坐标
        kpts3d = cameras.unproject_points(xy_depth)

        # 用valid将无效的点设置为全0
        kpts3d = kpts3d * valid.unsqueeze(-1)
        # 创建像素的网格
        x = torch.linspace(0, width - 1, width).repeat(height, 1).reshape(-1).to(self.device)
        y = torch.linspace(0, height - 1, height).repeat(width, 1).t().reshape(-1).to(self.device)

        # 对所有batch进行操作
        x = x.repeat(batchsize)
        y = y.repeat(batchsize)

        # 重塑深度图
        z = depth.reshape(batchsize * height * width)

        # 构建点云，每个点由x, y, z坐标表示
        xy_depth = torch.stack((x, y, z), dim=1)
        b = xy_depth.reshape(batchsize, height, width, 3)[0].detach().numpy()

        # 重塑为(Batch Size, N, 3)的形式
        xy_depth = xy_depth.reshape(batchsize, -1, 3)
        xyz = cameras.unproject_points(xy_depth)
        xyz_np = xyz[0].cpu().numpy()
        xyz_np = xyz_np.reshape(-1, 3)

        # 创建一个 Open3D PointCloud 对象
        pcd = o3d.geometry.PointCloud()

        # 使用 xyz 数据更新 PointCloud
        pcd.points = o3d.utility.Vector3dVector(xyz_np)

        # 可视化点云
        # o3d.visualization.draw_geometries([pcd])

        projected_xyz = self.perspective_projection(
            xyz,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batchsize, -1, -1),
            translation=torch.zeros(size=[batchsize, 3], device=self.device),
            focal_length=focal_length,
            camera_center=torch.Tensor([384 // 2, 216 // 2]).to(self.device).expand(batchsize, -1)
        )
        a = projected_xyz[0].detach().cpu().numpy().reshape(216, 384, 2)

        projected_xyz_human_0 = projected_xyz[0].reshape(216, 384, 2)[human_mask[0]].long()

        # projected_xyz_human_0xyz_np = xyz[0].cpu().numpy()
        # xyz_np = xyz_np.reshape(-1, 3)

        # 创建一个 Open3D PointCloud 对象
        # pcd = o3d.geometry.PointCloud()
        #
        # # 使用 xyz 数据更新 PointCloud
        # pcd.points = o3d.utility.Vector3dVector(xyz_np)
        #
        # # 可视化点云
        # o3d.visualization.draw_geometries([pcd])

        # 创建一个全零的RGB图像
        rendered_image = torch.zeros(height, width, 3)

        # 将对应的位置设置为红色
        rendered_image[projected_xyz_human_0[:, 1], projected_xyz_human_0[:, 0], 0] = 255

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(depth[0].detach().cpu().numpy(), cmap='gray')
        plt.title("Original depth image")

        plt.subplot(1, 2, 2)
        plt.imshow(rendered_image.detach().cpu().numpy())
        plt.title("Rendered image")
        plt.show()

    def get_pcd3(self, depth, focal_length, human_mask=None):
        batchsize, height, width = depth.shape
        xyz = self.depth_to_pointcloud(depth_map=depth,
                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batchsize, -1, -1),
                                       translation=torch.zeros(size=[batchsize, 3], device=self.device),
                                       focal_length=focal_length,
                                       camera_center=torch.Tensor([384 / 2, 216 / 2]).to(self.device).expand(batchsize,
                                                                                                             -1))
        xyz_np = xyz[0].cpu().numpy()
        xyz_np = xyz_np.reshape(-1, 3)

        # 创建一个 Open3D PointCloud 对象
        pcd = o3d.geometry.PointCloud()

        # 使用 xyz 数据更新 PointCloud
        pcd.points = o3d.utility.Vector3dVector(xyz_np)

        # 可视化点云
        # o3d.visualization.draw_geometries([pcd])

        # xyz_human = xyz[human_mask]

        # projected_xyz = self.perspective_projection(
        #     xyz.reshape(batchsize, -1, 3),
        #     rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batchsize, -1, -1),
        #     translation=torch.zeros(size=[batchsize, 3], device=self.device),
        #     focal_length=focal_length,
        #     camera_center=torch.Tensor([384 / 2, 216 / 2]).to(self.device).expand(batchsize, -1)
        # )
        # # projected_xyz = projected_xyz.reshape(batchsize, width, width, 2)
        # a = projected_xyz.detach().cpu().numpy()[0]
        #
        # projected_xyz_human_0 = projected_xyz[0][human_mask[0]]
        #
        #
        # projected_xyz_human_0 = projected_xyz_human_0.round().long()
        #
        # # 创建一个全零的RGB图像
        # rendered_image = torch.zeros(depth[0].size(0), depth[0].size(1), 3)
        #
        # # 将对应的位置设置为红色
        # rendered_image[projected_xyz_human_0[:, 0], projected_xyz_human_0[:, 1], 0] = 255
        #
        # plt.figure(figsize=(10, 5))
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(depth[0].detach().cpu().numpy(), cmap='gray')
        # plt.title("Original depth image")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(rendered_image.detach().cpu().numpy())
        # plt.title("Rendered image")
        # plt.show()


if __name__ == '__main__':
    from datasets.gta_im_all_10801920_plane_change_intri import GTADataset
    from torch.utils.data import DataLoader
    import numpy as np

    batchsize = 16
    dataset_train = GTADataset(data_dir='/Users/lwz/torch_ds/gta-im-fixbug/FPS-5')
    gta_loader = DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
    trans_model = TransNet()
    for idx, batch in enumerate(gta_loader):
        leres_image = batch['leres_image']
        image_path = batch['image_path']
        focal_length = batch['focal_length']
        depth_gt = batch['depth']
        intrinsic = batch['intrinsic']
        human_mask = batch['human_mask']
        kpts_2d=batch['kpts_2d']
        # trans_model.get_pcd2(depth_gt, intrinsic, human_mask)
        trans_model.get_pcd_kpts2d(depth_gt, intrinsic, kpts_2d)
        # trans_model.get_pcd3(depth_gt, focal_length, human_mask)

        if idx == 10:
            break
