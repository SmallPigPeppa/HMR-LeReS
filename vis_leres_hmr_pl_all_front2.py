import cv2
import os
import numpy as np
import open3d as o3d
import trimesh
import pyrender
from pyrender import Node, DirectionalLight
from leres_model_pl import LeReS
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import  tqdm

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



def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def main(index):
    # index = 559
    out_path = 'leres-vis-out-all-2'
    data_path='C:\\Users\\90532\\Desktop\\Datasets\\HMR-LeReS\\2020-06-11-10-06-48'
    mesh_path=os.path.join('C:\\Users\\90532\\Desktop\\Datasets\\GTA-IM\\meshes\\',f'{index+1:03d}.obj')
    info_pkl = np.load(os.path.join(data_path, 'info_frames.pickle'), allow_pickle=True)
    info_npz = np.load(os.path.join(data_path, 'info_frames.npz'))
    image_path=os.path.join(data_path,f'{index:05d}.jpg')
    focal_length = info_npz['intrinsics'][index, 0, 0]


    pl_ckpt_path = 'leres-ckpt-v4.0-debug/last-v4.ckpt'
    leres_model = LeReS.load_from_checkpoint(pl_ckpt_path)
    depth_model = leres_model.depth_model.eval()

    rgb = cv2.imread(image_path)
    rgb_resize = cv2.resize(rgb[:, :, ::-1], (448, 448))

    img_torch = scale_torch(rgb_resize)[None, :, :, :]
    depth, _ = depth_model(img_torch)
    depth = depth.cpu().detach().numpy().squeeze()
    dmax = np.percentile(depth, 90)
    dmin = np.percentile(depth, 10)
    depth = np.maximum(depth, dmin)
    depth = np.minimum(depth, dmax)
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

    # pyrender.Viewer(scene)

    cam_t = np.array([0, 0, -2]).reshape([3, 1])
    cam_rotate = pcd.get_rotation_matrix_from_xyz((np.pi * 1.0, np.pi * 0, np.pi * 0))
    camera_pose = np.hstack([cam_rotate, cam_t])
    camera_pose = np.vstack([camera_pose, [0, 0, 0, 1]])
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1.08137000e+03, fy=1.08137000e+03,
        cx=9.59500000e+02, cy=5.39500000e+02, zfar=10e20
    )
    scene.add(camera, pose=camera_pose)
    # pyrender.Viewer(scene)
    r = pyrender.OffscreenRenderer(
        viewport_width=1920,
        viewport_height=1080,
        point_size=1.0
    )
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # plt.imshow(color)
    # plt.show()
    im = Image.fromarray(color)
    im.save(os.path.join(out_path, f"{index:05d}.png"))



if __name__ == '__main__':
    for i in tqdm(range(0,955)):
        main(i)


