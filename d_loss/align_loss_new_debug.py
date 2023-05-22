import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from kornia.contrib import distance_transform


class AlignLoss(pl.LightningModule):
    def __init__(self):
        super(AlignLoss, self).__init__()
        self.kernal_size = 7
        self.edt_power = 0.5

    def convert_intrinsic(self, intrinsic):
        batchsize = intrinsic.shape[0]
        intrinsic_4x4 = torch.zeros(batchsize, 4, 4, dtype=intrinsic.dtype, device=self.device)
        intrinsic_4x4[:, :3, :3] = intrinsic
        intrinsic_4x4[:, 2, 3] = 1
        intrinsic_4x4[:, 3, 2] = 1
        return intrinsic_4x4

    def init_pytorch3d_render(self, intrinsic, image_size):
        cam_rotate = torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]], dtype=intrinsic.dtype, device=self.device)
        cam_trans = torch.zeros((1, 3), dtype=intrinsic.dtype, device=self.device)
        intrinsic_4x4 = self.convert_intrinsic(intrinsic)
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 0.0]])
        cameras = PerspectiveCameras(device=self.device, R=cam_rotate, T=cam_trans, K=intrinsic_4x4,
                                     image_size=[image_size], in_ndc=False)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=10,
        )
        self.renderer_pytorch3d = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )

    def get_depth_map_pytorch3d(self, verts, faces):
        batchsize = verts.shape[0]
        texture = torch.tensor([0.9, 0.7, 0.7], device=self.device).repeat(batchsize, verts.shape[1], 1)
        faces_batch = faces.repeat(batchsize, 1, 1)
        texture_pt3d = TexturesVertex(verts_features=texture)
        meshes_screen = Meshes(verts=verts, faces=faces_batch, textures=texture_pt3d)
        images, fragments = self.renderer_pytorch3d(meshes_screen)
        pix_to_face, zbuf, bary_coords, dists = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords, fragments.dists
        vismask = (pix_to_face[..., 0] > -1).bool()
        nearest_depth_map = zbuf.clone()
        farrest_depth_map = zbuf.clone()
        farrest_depth_map = farrest_depth_map.max(-1)[0]
        farrest_depth_map[~vismask] = 0.0
        nearest_depth_map[zbuf == -1] = 10.0
        nearest_depth_map = nearest_depth_map.min(-1)[0]
        nearest_depth_map[~vismask] = 0.0
        return vismask, nearest_depth_map, farrest_depth_map

    def distance_transform_manhattan(self, image):
        dist = distance_transform(image, kernel_size=3, h=0.35)
        return dist

    def distance_transform_euclidean(self, image):
        euclidean_dist = distance_transform_edt(1 - image.cpu().numpy())
        dist = torch.tensor(euclidean_dist, dtype=torch.float32, device=self.device)
        return dist

    def forward(self, faces, verts_gt, verts_pred, depth_pred, batch):
        intrinsic = batch['intrinsic']
        image_size = (1080 // 5, 1920 // 5)
        intrinsic_scaled = intrinsic
        # intrinsic_scaled = intrinsic / 5.0
        # intrinsic_scaled[2, 2] = 1  # Restore the [2, 2] element back to 1
        human_mask = batch['human_mask']
        self.init_pytorch3d_render(intrinsic=intrinsic_scaled, image_size=image_size)
        mesh_mask_gt, nearest_depth_map_gt, farrest_depth_map_gt = self.get_depth_map_pytorch3d(verts_gt, faces)
        mesh_mask_pred, nearest_depth_map_pred, farrest_depth_map_pred = self.get_depth_map_pytorch3d(verts_pred, faces)

        # human depth align loss
        human_mesh_mask_gt = torch.logical_and(mesh_mask_gt, human_mask)
        leres_human_depth = depth_pred[human_mesh_mask_gt]
        hmr_huamn_depth = nearest_depth_map_pred[human_mesh_mask_gt]
        human_depth_loss = F.l1_loss(leres_human_depth, hmr_huamn_depth)

        # mesh project align loss
        mesh_mask_gt_float = mesh_mask_gt.to(float).unsqueeze(1)
        mesh_mask_pred_float = mesh_mask_pred.to(float).unsqueeze(1)
        mesh_edge_gt = F.max_pool2d(mesh_mask_gt_float, self.kernal_size, 1, self.kernal_size // 2) - mesh_mask_gt_float
        mesh_edge_pred = F.max_pool2d(mesh_mask_pred_float, self.kernal_size, 1,
                                      self.kernal_size // 2) - mesh_mask_pred_float
        mesh_edge_dist_gt = self.distance_transform_manhattan(mesh_edge_gt)
        mesh_edge_dist_gt = mesh_edge_dist_gt ** self.edt_power
        # mesh_edge_dist_gt2= self.distance_transform_euclidean(mesh_edge_gt)
        # mesh_edge_dist_gt2 = mesh_edge_dist_gt2 ** self.edt_power
        mesh_project_edge_loss = torch.mean(mesh_edge_dist_gt * mesh_edge_pred)
        mash_project_loss = F.mse_loss(mesh_mask_gt_float, mesh_mask_pred_float)

        return {'human_depth_loss': human_depth_loss, 'mash_project_loss': mash_project_loss,
                'mesh_project_edge_loss': mesh_project_edge_loss}
        # return mesh_mask_gt,  mesh_edge_dist_gt,mesh_edge_dist_gt2

    def vis_forward(self, faces, verts_gt, verts_pred, depth_pred, batch):
        intrinsic = batch['intrinsic']
        image_size = (1080 // 5, 1920 // 5)
        intrinsic_scaled = intrinsic
        # intrinsic_scaled = intrinsic / 5.0
        # intrinsic_scaled[2, 2] = 1  # Restore the [2, 2] element back to 1
        human_mask = batch['human_mask']
        self.init_pytorch3d_render(intrinsic=intrinsic_scaled, image_size=image_size)
        mesh_mask_gt, nearest_depth_map_gt, farrest_depth_map_gt = self.get_depth_map_pytorch3d(verts_gt, faces)
        mesh_mask_pred, nearest_depth_map_pred, farrest_depth_map_pred = self.get_depth_map_pytorch3d(verts_pred, faces)

        # human depth align loss
        human_mesh_mask_gt = torch.logical_and(mesh_mask_gt, human_mask)
        leres_human_depth = depth_pred[human_mesh_mask_gt]
        hmr_huamn_depth = nearest_depth_map_pred[human_mesh_mask_gt]
        human_depth_loss = F.l1_loss(leres_human_depth, hmr_huamn_depth)

        # mesh project align loss
        mesh_mask_gt_float = mesh_mask_gt.to(float).unsqueeze(1)
        mesh_mask_pred_float = mesh_mask_pred.to(float).unsqueeze(1)
        mesh_edge_gt = F.max_pool2d(mesh_mask_gt_float, self.kernal_size, 1, self.kernal_size // 2) - mesh_mask_gt_float
        mesh_edge_pred = F.max_pool2d(mesh_mask_pred_float, self.kernal_size, 1,
                                      self.kernal_size // 2) - mesh_mask_pred_float
        mesh_edge_dist_gt = self.distance_transform_manhattan(mesh_edge_gt)
        mesh_edge_dist_gt = mesh_edge_dist_gt ** self.edt_power
        mesh_edge_dist_gt2 = self.distance_transform_euclidean(mesh_edge_gt)
        mesh_edge_dist_gt2 = mesh_edge_dist_gt2 ** self.edt_power
        mesh_project_edge_loss = torch.mean(mesh_edge_dist_gt * mesh_edge_pred)
        mash_project_loss = F.mse_loss(mesh_mask_gt_float, mesh_mask_pred_float)

        # return {'human_depth_loss': human_depth_loss, 'mash_project_loss': mash_project_loss,
        #         'mesh_project_edge_loss': mesh_project_edge_loss}
        return mesh_mask_gt, mesh_edge_dist_gt, mesh_edge_dist_gt2

    # def vis_human_depth_align_loss(self, verts, faces, depth, batch, pred_kpts_2d, pred_kpts_3d):
    #     loss = 0.
    #     batchsize = batch['leres_image'].shape[0]
    #     intrinsic = batch['intrinsic']
    #     intrinsic_scaled = intrinsic / 5.0
    #     intrinsic_scaled[2, 2] = 1  # Restore the [2, 2] element back to 1
    #     human_mask = batch['human_mask']
    #     self.init_pytorch3d_render(intrinsic=intrinsic, image_size=(1080 // 5, 1920 // 5))
    #     self.init_pytorch3d_render(intrinsic=intrinsic_scaled, image_size=(1080 // 5, 1920 // 5))
    #     mesh_mask, nearest_depth_map, farrest_depth_map = self.get_depth_map_pytorch3d(verts, faces)
    #     # human_mesh_mask = torch.logical_and(mesh_mask, human_mask)
    #     human_mesh_mask = mesh_mask
    #
    #     leres_human_depth = depth[human_mesh_mask]
    #     hmr_huamn_depth = nearest_depth_map[human_mesh_mask]
    #
    #     align_loss = torch.abs(leres_human_depth - hmr_huamn_depth)
    #     align_loss = torch.mean(align_loss)
    #
    #     for i in range(batchsize):
    #         import matplotlib.pyplot as plt
    #         import numpy as np
    #         leres_image = batch['leres_image'][i]
    #         hmr_image = batch['hmr_image'][i]
    #         kpts_2d = pred_kpts_2d[i]
    #         leres_image[~(human_mesh_mask[i].repeat(3, 1, 1))] *= 0
    #         f, axarr = plt.subplots(1, 2)
    #         plt.figure()
    #         leres_image = leres_image.permute(1, 2, 0)
    #         hmr_image = hmr_image.permute(1, 2, 0)
    #         axarr[0].imshow(leres_image)
    #         axarr[1].imshow(hmr_image)
    #         for j in range(24):
    #             axarr[0].scatter(np.squeeze(kpts_2d)[j][0], np.squeeze(kpts_2d)[j][1], s=50, c='red', marker='o')
    #         plt.show()
    #
    #     return loss

    # def mesh_project_align_loss(self, gt_mask, pred_mask, valid_mask=None):
    #     gt_mask_edge = F.max_pool2d(gt_mask, self.kernal_size, 1, self.kernal_size // 2) - gt_mask
    #     pred_mask_edge = F.max_pool2d(pred_mask, self.kernal_size, 1, self.kernal_size // 2) - pred_mask
    #
    #     gt_edge_dist = self.distance_transform_manhattan(gt_mask_edge)
    #     # gt_edge_dist = self.distance_transform_euclidean(gt_mask_edge)
    #     gt_edge_dist = gt_edge_dist ** self.edt_power
    #
    #     mask_edge_loss = torch.mean(pred_mask_edge * gt_edge_dist)
    #     mask_l2_loss = F.mse_loss(gt_mask, pred_mask)
    #     return mask_edge_loss, mask_l2_loss
    #
    # def human_depth_align_loss(self, verts, faces, depth, batch, ):
    #     intrinsic = batch['intrinsic']
    #     intrinsic_scaled = intrinsic / 5.0
    #     intrinsic_scaled[2, 2] = 1  # Restore the [2, 2] element back to 1
    #
    #     human_mask = batch['human_mask']
    #     self.init_pytorch3d_render(intrinsic=intrinsic, image_size=(1080 // 5, 1920 // 5))
    #     self.init_pytorch3d_render(intrinsic=intrinsic_scaled, image_size=(1080 // 5, 1920 // 5))
    #     mesh_mask, nearest_depth_map, farrest_depth_map = self.get_depth_map_pytorch3d(verts, faces)
    #     human_mesh_mask = torch.logical_and(mesh_mask, human_mask)
    #     # human_mesh_mask = mesh_mask
    #
    #     leres_human_depth = depth[human_mesh_mask]
    #     hmr_huamn_depth = nearest_depth_map[human_mesh_mask]
    #
    #     align_loss = F.l1_loss(leres_human_depth, hmr_huamn_depth)
    #     return align_loss
