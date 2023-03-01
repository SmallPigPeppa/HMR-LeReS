import cv2
import os
import numpy as np
import open3d as o3d
import trimesh
import pyrender
from pyrender import Node, DirectionalLight
from leres_model_pl import LeReS

def create_raymond_lights():
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


if __name__ == '__main__':
    focal_length = 1.15803374e+03
    out_path = 'leres-vis-out'
    image_path = os.path.join(out_path, '00559.jpg')
    data_path='C:\\Users\\90532\\Desktop\\Datasets\\HMR-LeReS\\2020-06-11-10-06-48'
    mesh_path='C:\\Users\\90532\\Desktop\\Datasets\\GTA-IM\\meshes\\559.obj'
    info_pkl = np.load(os.path.join(data_path, 'info_frames.pickle'), allow_pickle=True)
    info_npz = np.load(os.path.join(data_path, 'info_frames.npz'))


    leres_model = LeReS()
    depth_model = leres_model.depth_model.eval()
    leres_loader = leres_model.train_dataloader()['leres_loader']
    depth = next(iter(leres_loader))['depth'].numpy().squeeze()

    dmax = np.percentile(depth, 90)
    dmin = np.percentile(depth, 10)
    depth = np.maximum(depth, dmin)
    depth = np.minimum(depth, dmax)
    depth = np.minimum(depth, 15)

    depth_ori = cv2.resize(depth, (1920, 1080),interpolation=cv2.INTER_NEAREST)
    depth_ori = o3d.geometry.Image(depth_ori.astype(np.float32))
    color_raw = o3d.io.read_image(image_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_ori,
        depth_scale=1.0,
        depth_trunc=100.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsic(
                1920, 1080, focal_length, focal_length, 960.0, 540.0
            )
        ),
    )
    depth_pts = np.asarray(pcd.points)
    # depth_pts_aug = np.hstack(
    #     [depth_pts, np.ones([depth_pts.shape[0], 1])]
    # )
    # cam_extr_ref = np.linalg.inv(info_npz['world2cam_trans'][i])
    # depth_pts = depth_pts_aug.dot(cam_extr_ref)[:, :3]
    pcd.points = o3d.utility.Vector3dVector(depth_pts)
    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points.extend(pcd.points)
    global_pcd.colors.extend(pcd.colors)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    pcd_translate = np.array([[0, 0, 0]]).T
    pcd_rotate = np.identity(3)
    pcd_pose = np.hstack([pcd_rotate, pcd_translate])
    pcd_pose = np.vstack([pcd_pose, [0, 0, 0, 1]])
    mesh_pcd = pyrender.Mesh.from_points(points=np.array(pcd.points), colors=np.array(pcd.colors), poses=pcd_pose)
    scene.add(mesh_pcd, 'mesh-pcd', pose=pcd_pose)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    trimesh_human = trimesh.load(mesh_path)
    mesh_human = pyrender.Mesh.from_trimesh(trimesh_human, material=material)
    mesh_pose = np.eye(4)
    scene.add(mesh_human, 'mesh-human', pose=mesh_pose)


    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    pyrender.Viewer(scene)


