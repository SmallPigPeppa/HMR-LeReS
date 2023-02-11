import pytorch_lightning as pl

import torch
import pytorch3d
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
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

import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image


class AlignLoss(pl.LightningModule):
    def __init__(self, K_intrin, image_size):
        super(AlignLoss, self).__init__()
        self.K_intrin = K_intrin
        self.image_size = image_size
        self.init_pytorch3d_render(K_intrin, image_size)

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

    def align_loss(self, verts, faces, depth):
        vismask, nearest_depth_map, farrest_depth_map = self.get_depth_map_pytorch3d(verts, faces)
