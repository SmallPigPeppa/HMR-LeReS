import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import pickle

import pytorch3d.renderer
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from hmr_leres_config import args
from datasets.hmr_data_utils import get_torch_image_cut_box, collect_valid_kpts, calc_aabb, off_set_scale_kpts
from datasets.leres_data_utils import read_depthmap, read_human_mask, pil_loader
from PIL import Image


class GTADataset(Dataset):
    def __init__(self, data_dir, scale_range=[1.2, 1.4], use_flip=False, flip_prob=0.3):
        self.data_dir = data_dir
        self.hmr_scale_range = [1.3, 1.5]
        self.leres_scale_range = [2.5, 5.0]
        self.use_flip = use_flip
        self.flip_prob = flip_prob
        self.hmr_size = 224
        self.leres_size = 448
        self.hmr_transforms = T.Compose([T.Resize((self.hmr_size, self.hmr_size)), T.ToTensor()])
        self.leres_transforms = T.Compose([T.Resize((self.leres_size, self.leres_size)), T.ToTensor()])
        # self.leres_transforms = T.Compose([T.Resize((1080, 1920)), T.ToTensor()])
        self.depth_transforms = T.Compose(
            [T.Resize((self.leres_size, self.leres_size), interpolation=InterpolationMode.NEAREST),
             T.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)))])
        self.mask_transforms = T.Compose(
            [T.Resize((self.leres_size, self.leres_size), interpolation=InterpolationMode.NEAREST),
             T.Lambda(lambda image: torch.from_numpy(np.array(image).astype(bool)))])
        self.load_data_set()

    def load_data_set(self):
        print('start loading gta-im data.')
        self.info_npz = np.load(os.path.join(self.data_dir, 'info_frames.npz'))
        self.info_pkl = np.load(os.path.join(self.data_dir, 'info_frames.pickle'), allow_pickle=True)
        self.info_hmr = np.load(os.path.join(self.data_dir, 'info_hmr.pickle'), allow_pickle=True)
        # self.info_hmr = pickle.load(open(os.path.join(self.data_dir, 'info_hmr.pickle'), 'rb'))
        # self.info_pkl = pickle.load(open(os.path.join(self.data_dir, 'info_frames.pickle'),  'rb'))

        self.image_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.boxs = []
        self.kpts_2d = []
        self.kpts_3d = []
        self.joints_2d = []
        self.joints_3d = []
        self.transls = []
        self.shapes = []
        self.poses = []
        self.cam_focal_length = []
        self.cam_near_clips = []
        self.cam_far_clips = []
        self.gta_heads_2d = []

        total_kpts2d = np.array(self.info_hmr['kpts_2d'])
        total_kpts3d = np.array(self.info_hmr['kpts_3d'])
        total_joints2d = np.array(self.info_hmr['joints_2d'])
        total_joints3d = np.array(self.info_hmr['joints_3d'])
        total_shape = np.array(self.info_hmr['shape'])
        total_pose = np.array(self.info_hmr['pose'])
        total_transl = np.array(self.info_hmr['transl'])
        total_gta_heads_2d = np.array(self.info_hmr['gta_heads_2d'])

        # assert len(total_kp2d) == len(total_kp3d) and \
        #        len(total_kp2d) == len(total_shape) and len(total_kp2d) == len(total_pose)

        self.num_samples = len(total_pose)

        for idx in range(self.num_samples):
            # kpts2d_i = total_kpts2d[idx].reshape((-1, 3))
            # lt, rb, v = calc_aabb(collect_valid_kpts(kpts2d_i))

            joints2d_i = total_joints2d[idx].reshape((-1, 3))
            lt, rb = calc_aabb(collect_valid_kpts(joints2d_i))
            self.boxs.append((lt, rb))  # left-top, right-bottom
            self.kpts_2d.append(total_kpts2d[idx].copy())
            self.kpts_3d.append(total_kpts3d[idx].copy())
            self.joints_2d.append(total_joints2d[idx].copy())
            self.joints_3d.append(total_joints3d[idx].copy())
            self.shapes.append(total_shape[idx].copy())
            self.poses.append(total_pose[idx].copy())
            self.transls.append(total_transl[idx].copy())
            self.gta_heads_2d.append(total_gta_heads_2d[idx].copy())

            img_path_i = os.path.join(self.data_dir, '{:05d}'.format(idx) + '.jpg')
            mask_path_i = os.path.join(self.data_dir, '{:05d}'.format(idx) + '_id.png')
            depth_path_i = os.path.join(self.data_dir, '{:05d}'.format(idx) + '.png')
            assert os.path.exists(img_path_i) and os.path.exists(mask_path_i) and os.path.exists(depth_path_i)
            self.image_paths.append(img_path_i)
            self.mask_paths.append(mask_path_i)
            self.depth_paths.append(depth_path_i)

            info_i = self.info_pkl[idx]
            cam_near_clip_i = info_i['cam_near_clip']
            if 'cam_far_clip' in info_i.keys():
                cam_far_clip_i = info_i['cam_far_clip']
            else:
                cam_far_clip_i = 800.
            self.cam_near_clips.append(cam_near_clip_i)
            self.cam_far_clips.append(cam_far_clip_i)

        print('finished load gta-im data, total {} samples'.format(self.num_samples))
        self.shape_average = np.average(self.shapes, axis=0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if index == 61:
            index = 60
        image_path = self.image_paths[index]
        box = self.boxs[index]
        kpts_2d = self.kpts_2d[index]
        kpts_3d = self.kpts_3d[index]
        joints_2d = self.joints_2d[index]
        joints_3d = self.joints_3d[index]
        gta_head_2d = self.gta_heads_2d[index]

        # leres
        # origin_image = cv2.imread(image_path)
        # leres_image = Image.fromarray(origin_image)
        origin_image = pil_loader(image_path)
        info_npz = np.load(os.path.join(self.data_dir, 'info_frames.npz'))
        intrinsic = info_npz['intrinsics'][index]
        focal_length = np.array(intrinsic[0][0]).astype(np.float32)
        depth = read_depthmap(self.depth_paths[index], self.cam_near_clips[index], self.cam_far_clips[index])
        depth = Image.fromarray(depth)
        human_mask = read_human_mask(self.mask_paths[index], gta_head_2d)
        human_mask = Image.fromarray(human_mask)

        # crop and rescale leres_image,depth,human_mask,kpts2d,joints2d,human_mask
        leres_scale = np.random.rand(4) * (self.leres_scale_range[1] - self.leres_scale_range[0]) + \
                      self.leres_scale_range[0]
        top, left, height, width = get_torch_image_cut_box(leftTop=box[0], rightBottom=box[1], ExpandsRatio=leres_scale)
        leres_cut_box = np.array([top, left, height, width])
        leres_image = T.functional.crop(origin_image, top, left, height, width)
        depth = T.functional.crop(depth, top, left, height, width)
        human_mask = T.functional.crop(human_mask, top, left, height, width)

        kpts_2d = off_set_scale_kpts(kpts_2d, left=left, top=top, height_ratio=self.leres_size / height,
                                     width_ratio=self.leres_size / width)
        joints_2d_origin = joints_2d.copy()
        joints_2d = off_set_scale_kpts(joints_2d, left=left, top=top, height_ratio=self.leres_size / height,
                                       width_ratio=self.leres_size / width)

        # hmr
        hmr_scale = np.random.rand(4) * (self.hmr_scale_range[1] - self.hmr_scale_range[0]) + self.hmr_scale_range[0]
        hmr_top, hmr_left, hmr_height, hmr_width = get_torch_image_cut_box(leftTop=box[0], rightBottom=box[1],
                                                                           ExpandsRatio=hmr_scale)
        hmr_image = torchvision.transforms.functional.crop(origin_image, hmr_top, hmr_left, hmr_height, hmr_width)
        # transl, shape, pose = self.transls[index], self.shapes[index], self.poses[index]

        transl, shape, pose = self.transls[index], self.shape_average, self.poses[index]
        theta = np.concatenate((transl, pose, shape), axis=0)

        boody_pose = pose[3:]
        global_pose = pose[:3]

        return {
            'leres_image': self.leres_transforms(leres_image),
            'hmr_image': self.hmr_transforms(hmr_image),
            'depth': self.depth_transforms(depth),
            'human_mask': self.mask_transforms(human_mask),
            'kpts_2d': torch.from_numpy(kpts_2d).float(),
            'kpts_3d': torch.from_numpy(kpts_3d).float(),
            'joints_2d': torch.from_numpy(joints_2d).float(),
            'joints_3d': torch.from_numpy(joints_3d).float(),
            'joints_2d_origin': torch.from_numpy(joints_2d_origin).float(),
            'theta': torch.from_numpy(theta).float(),
            'body_pose': torch.from_numpy(boody_pose).float(),
            'global_pose': torch.from_numpy(global_pose).float(),
            'focal_length': torch.from_numpy(focal_length).float(),
            'intrinsic': torch.from_numpy(intrinsic).float(),
            'leres_cut_box': torch.from_numpy(leres_cut_box)
        }

if __name__ == '__main__':
    data_dir = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48-add-transl'
    gta_dataset = GTADataset(data_dir)
    # for idx,i in enumerate(iter(gta_loader)):
    #     if i['kp_2d'].shape!=i['kp_3d'].shape:
    #         print('frame{idx}'.format(idx=idx))
    #         print(i['kp_2d'].shape,i['kp_3d'].shape)
    for idx, i in enumerate(iter(gta_dataset)):
        # if len(i['kp_2d']) != 23:
        #     print('frame{idx}'.format(idx=idx))
        #     print(i['kp_2d'].shape, i['kp_3d'].shape)
        leres_image = i['leres_image']
        hmr_image = i['hmr_image']
        kpts_2d = i['kpts_2d']

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
        # break