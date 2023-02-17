import torch
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from hmr_leres_config import args
from d_loss.utils import crop_and_resize




class AlignLoss(pl.LightningModule):
    def __init__(self):
        super(AlignLoss, self).__init__()

    def init_pytorch3d_render(self, K_intrin, image_size):
        R = torch.Tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]], device=self.device)
        T = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        K_intrin_pt3d = torch.zeros((1, 4, 4), dtype=torch.float32, device=self.device)
        K_intrin_pt3d[0, 0, 0] = K_intrin[0, 0, 0]
        K_intrin_pt3d[0, 0, 2] = K_intrin[0, 0, 2]
        K_intrin_pt3d[0, 1, 1] = K_intrin[0, 1, 1]
        K_intrin_pt3d[0, 1, 2] = K_intrin[0, 1, 2]

        K_intrin_pt3d[0, 2, 3] = 1.0
        K_intrin_pt3d[0, 3, 2] = 1.0

        lights = PointLights(device=self.device, location=[[0.0, 0.0, 0.0]])
        cameras = PerspectiveCameras(device=self.device, R=R, T=T, K=K_intrin_pt3d, image_size=image_size, in_ndc=False)

        raster_settings = RasterizationSettings(
            image_size=image_size[0],
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
        texture = torch.Tensor([0.9, 0.7, 0.7], device=self.device).repeat(1, verts.shape[1], 1)
        texture_pt3d = TexturesVertex(verts_features=texture)
        meshes_screen = Meshes(verts=verts, faces=faces, textures=texture_pt3d)

        # import pdb;pdb.set_trace()
        images, fragments = self.renderer_pytorch3d(meshes_screen)
        # pix_to_face, zbuf, bary_coords, dists = fragments
        pix_to_face, zbuf, bary_coords, dists = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords, fragments.dists
        vismask = (pix_to_face[..., 0] > -1).bool()
        nearest_depth_map = zbuf.clone()
        farrest_depth_map = zbuf.clone()
        # import pdb;pdb.set_trace()
        farrest_depth_map = farrest_depth_map.max(-1)[0]
        farrest_depth_map[~vismask] = 0.0
        # farrest_depth_map = torch.flipud(farrest_depth_map[0])[None]

        nearest_depth_map[zbuf == -1] = 10.0
        nearest_depth_map = nearest_depth_map.min(-1)[0]
        nearest_depth_map[~vismask] = 0.0

        return vismask, nearest_depth_map, farrest_depth_map

    def batch_align_loss(self, verts, faces, depth, batch):
        loss = 0.
        batchsize = batch['leres_image'].shape[0]
        for idx in range(batchsize):
            cut_box_i = batch['leres_cut_box'][idx]
            K_intrin_i = batch['intrinsic'][idx]
            human_mask_i = batch['human_mask'][idx]
            verts_i = verts[idx]
            depth_i = depth[idx]
            loss += self.sample_align_loss(verts_i, faces, depth_i, cut_box_i, K_intrin_i, human_mask_i)
        return loss

    def sample_align_loss(self, verts, faces, depth, cut_box, K_intrin, human_mask):

        leres_size = args.leres_size
        origin_size = args.origin_size

        # unzqueeze
        human_mask = human_mask.unsqueeze(0)
        verts = verts.unsqueeze(0)
        depth = depth.unsqueeze(0)
        K_intrin = K_intrin.unsqueeze(0)
        cut_box = cut_box
        faces = faces

        self.init_pytorch3d_render(K_intrin=K_intrin, image_size=[origin_size])
        mesh_mask, nearest_depth_map, farrest_depth_map = self.get_depth_map_pytorch3d(verts, faces)

        mesh_mask = crop_and_resize(mesh_mask, cut_box=cut_box, img_size=leres_size)
        vismask = torch.logical_and(mesh_mask, human_mask)

        nearest_depth_map = crop_and_resize(nearest_depth_map, cut_box=cut_box, img_size=leres_size)
        leres_human_depth = depth[vismask]
        hmr_huamn_depth = nearest_depth_map[vismask]

        align_loss = torch.abs(leres_human_depth - hmr_huamn_depth)
        align_loss = torch.mean(align_loss)
        return align_loss

    def vis_batch_align_loss(self, verts, faces, depth, batch):
        loss = 0.
        batchsize = batch['leres_image'].shape[0]
        for idx in range(batchsize):
            cut_box_i = batch['leres_cut_box'][idx]
            K_intrin_i = batch['intrinsic'][idx]
            leres_image_i = batch['leres_image'][idx]
            hmr_image_i = batch['hmr_image'][idx]
            kpts_2d_i = batch['kpts_2d'][idx]
            kpts_2d_i = batch['joints_2d'][idx]
            human_mask_i = batch['human_mask'][idx]
            verts_i = verts[idx]
            depth_i = depth[idx]
            loss += self.vis_sample_align_loss(verts_i, faces, depth_i, cut_box_i, K_intrin_i, human_mask_i,
                                               leres_image_i, hmr_image_i,
                                               kpts_2d_i)
        return loss

    def vis_sample_align_loss(self, verts, faces, depth, cut_box, K_intrin, human_mask, leres_image, hmr_image,
                              kpts_2d):

        leres_size = args.leres_size
        origin_size = args.origin_size

        # unzqueeze
        human_mask = human_mask.unsqueeze(0)
        verts = verts.unsqueeze(0)
        depth = depth.unsqueeze(0)
        K_intrin = K_intrin.unsqueeze(0)
        cut_box = cut_box
        faces = faces

        self.init_pytorch3d_render(K_intrin=K_intrin, image_size=[origin_size])
        mesh_mask, nearest_depth_map, farrest_depth_map = self.get_depth_map_pytorch3d(verts, faces)

        mesh_mask = crop_and_resize(mesh_mask, cut_box=cut_box, img_size=leres_size)
        vismask = torch.logical_and(mesh_mask, human_mask)
        # vismask = mesh_mask

        nearest_depth_map = crop_and_resize(nearest_depth_map, cut_box=cut_box, img_size=leres_size)
        leres_human_depth = depth[vismask]
        hmr_huamn_depth = nearest_depth_map[vismask]

        # align_loss = torch.mean(torch.abs(leres_human_depth - hmr_huamn_depth))
        align_loss = torch.abs(leres_human_depth - hmr_huamn_depth)
        align_loss = torch.mean(align_loss)

        # leres_image[~(vismask[0].repeat(3, 1, 1))] *= 0
        leres_image[~(human_mask[0].repeat(3, 1, 1))] *= 0

        import matplotlib.pyplot as plt
        import numpy as np

        f, axarr = plt.subplots(1, 2)
        plt.figure()
        leres_image = leres_image.permute(1, 2, 0)
        hmr_image = hmr_image.permute(1, 2, 0)
        axarr[0].imshow(leres_image)
        axarr[1].imshow(hmr_image)
        # plt.imshow(hmr_image.permute(1, 2, 0))
        for j in range(24):
            # plt.imshow(img)
            axarr[0].scatter(np.squeeze(kpts_2d)[j][0], np.squeeze(kpts_2d)[j][1], s=50, c='red', marker='o')
        plt.show()

        return align_loss
