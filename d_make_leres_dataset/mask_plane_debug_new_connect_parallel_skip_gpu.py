import os
import cv2
import numpy as np
# from sklearn.cluster import DBSCAN
from cuml.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import open3d as o3d
from skimage.measure import label
import argparse
import torch
import cuml




def read_depthmap(depth_path, cam_near_clip, cam_far_clip):
    try:
        depth = cv2.imread(depth_path)
        if depth is None:
            print(f"Warning: Failed to read file '{depth_path}'. Skipping.")
            return None
    except Exception as e:
        print(f"Warning: Failed to read file '{depth_path}'. Exception: {e}. Skipping.")
        return None

    depth = np.concatenate(
        (depth, np.zeros_like(depth[:, :, 0:1], dtype=np.uint8)), axis=2
    )
    depth.dtype = np.uint32
    depth = 0.05 * 1000 / (depth.astype('float') + 1e-5)
    depth = (
            cam_near_clip
            * cam_far_clip
            / (cam_near_clip + depth * (cam_far_clip - cam_near_clip))
    )
    return np.squeeze(depth)






def compute_normals(depth_img, intrinsic, window_size=5):
    h, w = depth_img.shape
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

    # 创建Open3D中的PinholeCameraIntrinsic对象
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # 将深度图像转换为float32数据类型
    depth_img_float = depth_img.astype(np.float32)

    # 将深度图像转换为Open3D中的Image对象
    depth_o3d = o3d.geometry.Image(depth_img_float)

    # 从深度图像和相机内参计算点云
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, pinhole_camera_intrinsic)

    # 计算法线
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=window_size, max_nn=30))

    # 将法线从Open3D格式转换回NumPy数组
    o3d_normals = np.asarray(point_cloud.normals)

    # 创建一个与深度图像相同形状的零数组
    normals = np.zeros((h, w, 3))

    # 获取深度图像中非零像素的索引
    non_zero_indices = np.nonzero(depth_img)

    # 将计算得到的法线值填充到相应的位置
    normals[non_zero_indices] = o3d_normals

    return normals


def filter_clusters_by_size(plane_masks, min_cluster_size):
    filtered_plane_masks = plane_masks.copy()
    unique_cluster_ids = np.unique(plane_masks)

    for cluster_id in unique_cluster_ids:
        if cluster_id == 0:  # 忽略背景值（0）
            continue
        cluster_size = np.sum(plane_masks == cluster_id)
        if cluster_size < min_cluster_size:
            filtered_plane_masks[plane_masks == cluster_id] = 0

    return filtered_plane_masks





def compute_plane_masks(depth_img, intrinsic, eps=0.02, min_samples=2000):
    normals = compute_normals(depth_img, intrinsic)
    h, w, _ = normals.shape
    normals_flat = normals.reshape(-1, 3)
    mask = depth_img.reshape(-1) != 0

    # Move data to GPU
    normals_flat_gpu = torch.tensor(normals_flat[mask], device="cuda:0", dtype=torch.float32)

    # Perform DBSCAN clustering using cuml
    # dbscan = cuml.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, output_type='int64')
    cluster_labels = dbscan.fit_predict(normals_flat_gpu)

    plane_masks = np.zeros((h * w), dtype=np.uint8)
    # plane_masks[mask] = cluster_labels.cpu().numpy() + 1
    # plane_masks[mask] = cluster_labels + 1
    plane_masks[mask] = (cluster_labels + 1).get()

    plane_masks = plane_masks.reshape((h, w))

    # 对每个平面分别进行连通组件分析，并更新平面掩码
    num_planes = np.max(plane_masks)
    for i in range(1, num_planes + 1):
        plane_mask_i = (plane_masks == i).astype(np.uint8)
        labels_im = label(plane_mask_i)

        # 更新平面掩码中的标签
        plane_masks[labels_im > 0] = labels_im[labels_im > 0] + num_planes * (i - 1)

    return plane_masks


# def compute_plane_masks(depth_img, intrinsic, eps=0.02, min_samples=2000):
#     normals = compute_normals(depth_img, intrinsic)
#     h, w, _ = normals.shape
#     normals_flat = normals.reshape(-1, 3)
#     mask = depth_img.reshape(-1) != 0
#
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(normals_flat[mask])
#
#     plane_masks = np.zeros((h * w), dtype=np.uint8)
#     plane_masks[mask] = dbscan.labels_ + 1
#     plane_masks = plane_masks.reshape((h, w))
#
#     # 对每个平面分别进行连通组件分析，并更新平面掩码
#     num_planes = np.max(plane_masks)
#     for i in range(1, num_planes + 1):
#         plane_mask_i = (plane_masks == i).astype(np.uint8)
#         labels_im = label(plane_mask_i)
#
#         # 更新平面掩码中的标签
#         plane_masks[labels_im > 0] = labels_im[labels_im > 0] + num_planes * (i - 1)
#
#     return plane_masks

def main(scene_id, eps=0.02, min_cluster_size=1000, n=5):
    num_samples = 0
    eps = 0.02
    min_cluster_size=1000
    n = 5  # 缩放因子
    data_dir = '/Users/lwz/torch_ds/gta-im/FPS-5'
    # data_dir = 'C:\\Users\\90532\\Desktop\\Datasets\\gta-im\\FPS-5'
    data_dir='/mnt/mmtech01/dataset/vision_text_pretrain/gta-im/FPS-5'
    plane_mask_vis_dir='plane_mask_vis'
    plane_mask_dir = 'plane_mask'

    scene_dirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d))])

    if scene_id < 0 or scene_id >= len(scene_dirs):
        print(f"Invalid scene_id: {scene_id}. Scene_id should be between 0 and {len(scene_dirs) - 1}. Exiting.")
        return

    scene_dir = scene_dirs[scene_id]
    # 在外部创建 plane_mask_vis 和 plane_mask 基础目录
    if not os.path.exists(plane_mask_vis_dir):
        os.makedirs(plane_mask_vis_dir)
    if not os.path.exists(plane_mask_dir):
        os.makedirs(plane_mask_dir)

    last_component = scene_dir.split(os.sep)[-1]

    # 为 plane_mask_vis_dir 和 plane_mask_dir 构建子目录路径
    plane_mask_vis_scenedir = os.path.join(plane_mask_vis_dir, last_component)
    plane_mask_scenedir = os.path.join(plane_mask_dir, last_component)

    # 如果子目录不存在，则创建它们
    if not os.path.exists(plane_mask_vis_scenedir):
        os.makedirs(plane_mask_vis_scenedir)
    if not os.path.exists(plane_mask_scenedir):
        os.makedirs(plane_mask_scenedir)

    info_npz = np.load(os.path.join(scene_dir, 'info_frames.npz'))
    info_pkl = np.load(os.path.join(scene_dir, 'info_frames.pickle'), allow_pickle=True)
    num_samples += len(info_pkl)
    scence_samples = len(info_pkl)

    colors = list(mcolors.CSS4_COLORS.values())
    num_colors = len(colors)
    colormap = mcolors.ListedColormap(colors[:num_colors])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))


    for idx in tqdm(range(scence_samples)):

        plane_mask_file = os.path.join(plane_mask_scenedir, '{:05d}'.format(idx) + '_plane.png')
        if os.path.exists(plane_mask_file):
            continue


        info_i = info_pkl[idx]
        intrinsic_i = info_npz['intrinsics'][idx].astype(np.float32)  # 确保内参矩阵为float32类型
        depth_path_i = os.path.join(scene_dir, '{:05d}'.format(idx) + '.png')

        cam_near_clip_i = info_i['cam_near_clip']
        if 'cam_far_clip' in info_i.keys():
            cam_far_clip_i = info_i['cam_far_clip']
        else:
            cam_far_clip_i = 800.

        depth_i = read_depthmap(depth_path_i, cam_near_clip_i, cam_far_clip_i)
        if depth_i is not None:
            # 缩放深度图像
            depth_i_resized = cv2.resize(depth_i, (1920 // n, 1080 // n), interpolation=cv2.INTER_AREA)

            # 修改内参矩阵以适应缩放后的图像
            intrinsic_i_resized = intrinsic_i.copy()
            intrinsic_i_resized[0, 0] /= n  # 缩放 fx
            intrinsic_i_resized[1, 1] /= n  # 缩放 fy
            intrinsic_i_resized[0, 2] /= n  # 缩放 cx
            intrinsic_i_resized[1, 2] /= n  # 缩放 cy

            depth_i_resized = depth_i_resized * (depth_i_resized >= 0) * (depth_i_resized <= 15)

            # 使用缩放后的深度图像和内参矩阵计算平面掩码
            plane_masks_i = compute_plane_masks(depth_i_resized, intrinsic_i_resized, eps, min_cluster_size)
            plane_masks_i = filter_clusters_by_size(plane_masks_i, min_cluster_size)

            depth_i = read_depthmap(depth_path_i, cam_near_clip_i, cam_far_clip_i)
            img_path_i = os.path.join(scene_dir, '{:05d}'.format(idx) + '.jpg')
            img_i = cv2.imread(img_path_i)
            img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB格式

            # 标准化深度图和平面掩码以便显示
            normalized_depth_i = cv2.normalize(depth_i, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # 使用 matplotlib 展示原图、深度图和平面掩码图像
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(img_i)
            ax1.set_title("Original Image")
            ax1.axis("off")

            ax2.imshow(normalized_depth_i, cmap='gray')
            ax2.set_title("Depth Image")
            ax2.axis("off")

            ax3.imshow(plane_masks_i, cmap=colormap)
            ax3.set_title("Plane Masks")
            ax3.axis("off")

            # 修改：正确打开和关闭文件以保存结果
            with open(os.path.join(plane_mask_vis_scenedir, '{:05d}'.format(idx) + '.png'), 'wb') as file:
                plt.savefig(file, dpi=300, bbox_inches='tight', format='png')
            # plt.show()
            # plt.close(fig)

            # plane_mask_i_uint16 = plane_masks_i.astype(np.uint16)
            #
            # # 将 plane_mask_i_uint16 保存为 PNG 文件
            # cv2.imwrite(os.path.join(plane_mask_scenedir, '{:05d}'.format(idx) + '_plane.png'), plane_mask_i_uint16)

            plane_mask_i_uint8 = plane_masks_i.astype(np.uint8)

            # 将 plane_mask_i_uint16 保存为 PNG 文件
            cv2.imwrite(os.path.join(plane_mask_scenedir, '{:05d}'.format(idx) + '_plane.png'), plane_mask_i_uint8)
            # import gc
            # gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a single scene.")
    parser.add_argument('--scene_id', type=int, default=0, help='Scene ID to process (0-based index)')
    args = parser.parse_args()

    main(args.scene_id)
