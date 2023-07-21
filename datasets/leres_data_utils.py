import os
import cv2
import numpy as np
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert("RGB")
    except OSError:
        return None


def test_read_img(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read file '{img_path}'. Skipping.")
            return None
    except Exception as e:
        print(f"Warning: Failed to read file '{img_path}'. Exception: {e}. Skipping.")
        return None
    return img

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


def read_human_mask(mask_path, gta_head_2d):
    try:
        sem_mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
        if sem_mask is None:
            print(f"Warning: Failed to read file '{mask_path}'. Skipping.")
            return None
    except Exception as e:
        print(f"Warning: Failed to read file '{mask_path}'. Exception: {e}. Skipping.")
        return None

    human_id = sem_mask[
        np.clip(int(gta_head_2d[1]), 0, 1079), np.clip(int(gta_head_2d[0]), 0, 1919)
    ]
    # human_id=37

    human_mask = sem_mask == human_id
    return human_mask


def read_plane_mask(plane_mask_path):
    try:
        plane_mask = cv2.imread(plane_mask_path, cv2.IMREAD_ANYDEPTH)
        if plane_mask is None:
            print(f"Warning: Failed to read file '{plane_mask_path}'. Skipping.")
            return None
    except Exception as e:
        print(f"Warning: Failed to read file '{plane_mask_path}'. Exception: {e}. Skipping.")
        return None

    return plane_mask
