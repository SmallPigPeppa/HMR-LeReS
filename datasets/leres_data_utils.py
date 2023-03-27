import cv2
import numpy as np
from PIL import Image
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
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


def read_human_mask(mask_path,gta_head_2d):
    sem_mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
    human_id = sem_mask[
        np.clip(int(gta_head_2d[1]), 0, 1079), np.clip(int(gta_head_2d[0]), 0, 1919)
    ]
    # human_id=37

    human_mask = sem_mask == human_id
    return human_mask