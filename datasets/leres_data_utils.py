import cv2
import numpy as np
def read_depthmap(depth_path, cam_near_clip, cam_far_clip):
    depth = cv2.imread(depth_path)
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


def read_human_mask(mask_path,kpts_2d):
    return None