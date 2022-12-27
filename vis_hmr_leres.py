# Author: Julien Fischer
#
# Remark: Most of the pyrender code is taken from SMPLify-X. However, since the original SMPLify-X visualization code resulted in
#         errors on my system, I decided to write a new visualization method that is employed _after_ the fitting pipeline.
import pyrender
import os
import numpy as np
import trimesh
import argparse
from PIL import Image as pil_img
import matplotlib.pyplot as plt
from pyrender import Node, DirectionalLight
from typing import Tuple, Union, List
from tqdm import tqdm
import shutil
import open3d as o3d
from PIL import Image

try:
    import cPickle as pickle
except ImportError:
    import pickle


def _create_raymond_lights():
    """Taken from pyrender source code"""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(Node(
            light=DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def read_pickle_file(file_location) -> dict:
    """Reads the pickle file and returns it as a dictionary"""
    with open(file_location, "rb") as file:
        pickle_dict = pickle.load(file, encoding="latin1")
    return pickle_dict


def load_mesh(mesh_location: str) -> pyrender.Mesh:
    """Loads a mesh with applied material from the specified file."""

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    body_mesh = trimesh.load(mesh_location)
    mesh = pyrender.Mesh.from_trimesh(
        body_mesh,
        material=material)

    return mesh


def get_scene_render(body_mesh: pyrender.Mesh,
                     pcd,
                     image_width: int,
                     image_height: int,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Renders the scene and returns the color and depth output"""

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(body_mesh, 'mesh')

    pcd_translate = np.array([[0, 0, 0]]).T
    # camera_translate=np.array([[0,0,1000]]).T
    pcd_rotate = pcd.get_rotation_matrix_from_xyz((0, 0, 0))
    pcd_pose = np.hstack([pcd_rotate, pcd_translate])
    pcd_pose = np.vstack([pcd_pose, [0, 0, 0, 1]])
    tmp_points = np.array(pcd.points)
    # tmp_points[:,2]*=0.5
    tmp_points *= 0.0009
    mesh_pcd = pyrender.Mesh.from_points(points=tmp_points, colors=np.array(pcd.colors), poses=pcd_pose)
    scene.add(mesh_pcd, 'mesh-pcd')

    light_nodes = _create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)
    # pyrender.Viewer(scene)
    # camera_translation[0] *= -1.0
    # camera_pose = np.eye(4)
    # camera_pose[:3, 3] = camera_translation

    camera_pose = np.eye(4)
    # camera_pose = RT
    camera_pose[1, :] = - camera_pose[1, :]
    camera_pose[2, :] = - camera_pose[2, :]

    camera_rotate = pcd.get_rotation_matrix_from_xyz((0, np.pi*1.0, np.pi*1.0))
    camera_translate = np.array([[0, 0, 0]]).T
    # camera_rotate = pcd.get_rotation_matrix_from_xyz((0, np.pi*-0.7, np.pi*0.9))
    # camera_translate = np.array([[-3.5, -0.3, 0.8]]).T
    # camera_rotate = pcd.get_rotation_matrix_from_xyz((0, np.pi*-0.9, np.pi*1.0))
    # camera_translate = np.array([[-1.5, -0.15, -0.5]]).T
    # camera_rotate = pcd.get_rotation_matrix_from_xyz((np.pi*-0.3, np.pi*1.0, np.pi*1.0))
    # camera_translate = np.array([[1.0,-7.0,-0.2]]).T

    camera_pose = np.hstack([camera_rotate, camera_translate])
    camera_pose = np.vstack([camera_pose, [0, 0, 0, 1]])



    '''
    1.08137000e+03
    0.
    9.59500000e+02
    0.
    1.08137000e+03
    5.39500000e+02
    0.
    0.
    1.
    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    '''

    # camera = pyrender.camera.IntrinsicsCamera(
    #    fx=camera_focal_length, fy=camera_focal_length,
    #    cx=camera_center[0], cy=camera_center[1]
    # )
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1.08137000e+03, fy=1.08137000e+03,
        cx=9.59500000e+02, cy=5.39500000e+02, zfar=10e20
    )
    scene.add(camera, pose=camera_pose)
    pyrender.Viewer(scene)
    light_nodes = _create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    # scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(
        viewport_width=image_width,
        viewport_height=image_height,
        point_size=1.0
    )
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # plt.imshow(color)
    # plt.show()
    return color, depth


def combine_scene_image(scene_rgba: Union[np.ndarray, List[np.ndarray]],
                        original_image) -> pil_img:
    """Combines the rendered scene with the original image and returns it as PIL.Image
    If a list of rendered scenes is given, they are all added to the original image.
    TODO: Strategy when bodies overlap (e.g. consider depth maps to infer which body should be shown)
    """
    # rewrite such that only lists are accepted and processed
    original_normalized = (np.asarray(original_image) / 255.0).astype(np.float32)
    if isinstance(scene_rgba, np.ndarray):
        scene_normalized = scene_rgba.astype(np.float32) / 255.0
        valid_mask = (scene_normalized[:, :, -1] > 0)[:, :, np.newaxis]

        output_image = (scene_normalized[:, :, :-1] * valid_mask +
                        (1 - valid_mask) * original_normalized)
        img = pil_img.fromarray((output_image * 255).astype(np.uint8))
        return img
    elif isinstance(scene_rgba, list):
        bodies = []
        valid_masks = []
        scenes_normalized = np.asarray(scene_rgba).astype(np.float32) / 255.0
        for scene in scenes_normalized:
            valid_mask = (scene[:, :, -1] > 0)[:, :, np.newaxis]
            bodies.append(scene[:, :, :-1] * valid_mask)
            valid_masks.append(valid_mask)
        body_mask = (np.sum(valid_masks, axis=0)[:, :, 0] > 0)[:, :, np.newaxis]
        # for now: prevent overflow when bodies overlap by clipping. Later: only show the one in front using depth map
        output_image = np.clip(np.sum(bodies, axis=0), 0, 1) + (1 - body_mask) * original_normalized
        img = pil_img.fromarray((output_image * 255).astype(np.uint8))
        return img


def load_image(path) -> pil_img:
    """Loads the image from the given path and returns it"""
    img = pil_img.open(path)
    return img


def main(args):
    pcds = [file for file in os.listdir(args.pcd_dir) if os.path.splitext(file)[1] in ['.ply']]
    os.makedirs(args.output, exist_ok=True)

    # visualize each image separately
    for pcd in tqdm(pcds, desc="Align LeReS and mover Processing"):
        # at the moment, only a single person is supported
        pcd_name = os.path.splitext(pcd)[0]
        body_mesh_name=pcd_name.replace('-pcd','')

        pcd_file = o3d.io.read_point_cloud(os.path.join(args.pcd_dir, pcd))
        # pcd_file = o3d.io.read_point_cloud('out0001-pcd.ply')
        body_mesh= load_mesh(f"{os.path.join(args.mesh_dir, body_mesh_name)}.obj")
        # body_mesh = load_mesh("000.obj_cam_CS.obj")

        scene_rgba, _ = get_scene_render(
            body_mesh=body_mesh,
            pcd=pcd_file,
            image_width=1920,
            image_height=1080,
        )
        if args.show_results:
            # overlayed.show()
            plt.imshow(scene_rgba)
            plt.gcf().canvas.manager.set_window_title(f"{pcd_name}")
            plt.axis('off')
            plt.show()

        if not args.no_save:
            # scene_rgba.save(os.path.join(args.output, f"{pcd_name}.png"))
            im = Image.fromarray(scene_rgba)
            im.save(os.path.join(args.output, f"{body_mesh_name}.png"))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizes the fitted SMPL meshes on the original images.")
    parser.add_argument("--mesh_dir", type=str, default="meshs",
                        help="Path to the SMPLify-X output folder that contains the meshes and pickle files.")
    parser.add_argument("--pcd_dir", type=str, default="pcds",
                        help="Path to the folder that contains the input images.")
    parser.add_argument("--output", type=str, default="hmr-leres-align",
                        help="Location where the resulting images should be saved at.")
    parser.add_argument('--show_results', action="store_true", help="Show the resulting overlayed images.")
    parser.add_argument('--no_save', action="store_true", help="Do not save the resulting overlayed images.")
    args = parser.parse_args()

    main(args)
