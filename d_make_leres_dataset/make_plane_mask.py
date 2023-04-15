import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


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


def compute_normals(depth_img, focal_length):
    h, w = depth_img.shape
    x_range = np.linspace(0, w - 1, w)
    y_range = np.linspace(0, h - 1, h)
    x, y = np.meshgrid(x_range, y_range)

    X = (x - w / 2) * depth_img / focal_length
    Y = (y - h / 2) * depth_img / focal_length

    dX = cv2.Sobel(X, cv2.CV_64F, 1, 0, ksize=5)
    dY = cv2.Sobel(Y, cv2.CV_64F, 0, 1, ksize=5)
    dZ = np.ones_like(depth_img) * -1

    normals = np.stack([dX, dY, dZ], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normals[depth_img == 0] = 0
    return normals


def compute_plane_masks(depth_img, focal_length, eps=0.1, min_samples=100):
    normals = compute_normals(depth_img, focal_length)
    h, w, _ = normals.shape
    normals_flat = normals.reshape(-1, 3)
    mask = depth_img.reshape(-1) != 0

    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(normals_flat[mask])

    plane_masks = np.zeros((h * w), dtype=np.uint8)
    plane_masks[mask] = dbscan.labels_ + 1
    plane_masks = plane_masks.reshape((h, w))

    return plane_masks


# 示例：
# depth_img = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE)
# focal_length = 525  # 请根据实际情况设置焦距
#
# plane_masks = compute_plane_masks(depth_img, focal_length)
# cv2.imwrite("plane_masks.png", plane_masks * (255 // np.max(plane_masks)))

if __name__ == '__main__':
    num_samples = 0
    data_dir = '/Users/lwz/torch_ds/gta-im/FPS-5'
    scene_dirs=[os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))]

    for scene_dir in scene_dirs:
        info_npz = np.load(os.path.join(scene_dir, 'info_frames.npz'))
        info_pkl = np.load(os.path.join(scene_dir, 'info_frames.pickle'), allow_pickle=True)
        num_samples += len(info_pkl)
        scence_samples = len(info_pkl)
        for idx in range(scence_samples):
            info_i = info_pkl[idx]
            intrinsic_i = info_npz['intrinsics'][idx]
            focal_length = np.array(intrinsic_i[0][0]).astype(np.float32)
            depth_path_i = os.path.join(scene_dir, '{:05d}'.format(idx) + '.png')

            cam_near_clip_i = info_i['cam_near_clip']
            if 'cam_far_clip' in info_i.keys():
                cam_far_clip_i = info_i['cam_far_clip']
            else:
                cam_far_clip_i = 800.

            depth_i = read_depthmap(depth_path_i, cam_near_clip_i, cam_far_clip_i)
            plane_masks_i = compute_plane_masks(depth_i, focal_length)
            cv2.imshow(plane_masks_i * (255 // np.max(plane_masks_i)))
            a = 1
            # cv2.imwrite("plane_masks.png", plane_masks * (255 // np.max(plane_masks)))

        print(f'finished load from {scene_dir}, total {scence_samples} samples')

    print(f'finished load all gta-im data from {data_dir}, total {num_samples} samples')
