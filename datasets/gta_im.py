import sys

sys.path.append('./src')
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from hmr_leres_config import args
from hmr_data_utils import get_torch_image_cut_box, collect_valid_kpts, calc_aabb, off_set_kpts, scale_kpts
from leres_data_utils import read_depthmap, read_human_mask, pil_loader
from PIL import Image


class GTA_Dataset(Dataset):
    def __init__(self, data_dir, scale_range=[1.2, 1.4], use_flip=False, flip_prob=0.3):
        self.data_dir = data_dir
        self.scale_range = scale_range
        self.use_flip = use_flip
        self.flip_prob = flip_prob
        self.hmr_size = 224
        self.leres_size = 448
        self.hmr_transforms = T.Compose([T.Resize((self.hmr_size, self.hmr_size)), T.ToTensor()])
        self.leres_transforms = T.Compose([T.Resize((self.leres_size, self.leres_size)), T.ToTensor()])
        self.depth_transforms = T.Compose(
            [T.Resize((self.leres_size, self.leres_size), interpolation=InterpolationMode.NEAREST),
             T.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)))])
        self.load_data_set()

    def load_data_set(self):
        print('start loading gta-im data.')
        self.info_pkl = np.load(os.path.join(self.data_dir, 'info_frames.pickle'), allow_pickle=True)
        self.info_npz = np.load(os.path.join(self.data_dir, 'info_frames.npz'))
        self.info_hmr = np.load(os.path.join(self.data_dir, 'info_hmr.pickle'), allow_pickle=True)

        self.image_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.boxs = []
        self.kpts_2d = []
        self.kpts_3d = []
        self.transls = []
        self.shapes = []
        self.poses = []
        self.cam_focal_length = []
        self.cam_near_clips = []
        self.cam_far_clips = []

        total_kp2d = np.array(self.info_hmr['kpts_2d'])
        total_kp3d = np.array(self.info_hmr['kpts_3d'])
        total_shape = np.array(self.info_hmr['shape'])
        total_pose = np.array(self.info_hmr['pose'])
        total_transl = np.array(self.info_hmr['transl'])

        # assert len(total_kp2d) == len(total_kp3d) and \
        #        len(total_kp2d) == len(total_shape) and len(total_kp2d) == len(total_pose)

        self.num_samples = len(total_pose)

        for idx in range(self.num_samples):
            kp2d = total_kp2d[idx].reshape((-1, 3))
            lt, rb, v = calc_aabb(collect_valid_kpts(kp2d))
            self.kpts_2d.append(np.array(kp2d.copy(), dtype=float))
            self.boxs.append((lt, rb))  # left-top, right-bottom
            self.kpts_3d.append(total_kp3d[idx].copy())
            self.shapes.append(total_shape[idx].copy())
            self.poses.append(total_pose[idx].copy())
            self.transls.append(total_transl[idx].copy())

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # index = 88
        image_path = self.image_paths[index]
        kpts_2d = self.kpts_2d[index]
        box = self.boxs[index]
        kpts_3d = self.kpts_3d[index]

        # leres
        # origin_image = cv2.imread(image_path)
        # leres_image = Image.fromarray(origin_image)
        origin_image = pil_loader(image_path)
        leres_image = origin_image  # image aug
        info_npz = np.load(os.path.join(self.data_dir, 'info_frames.npz'))
        intrinsic = info_npz['intrinsics'][index]
        focal_length = np.array(intrinsic[0][0]).astype(np.float32)
        depth = read_depthmap(self.depth_paths[index], self.cam_near_clips[index], self.cam_far_clips[index])
        depth = Image.fromarray(depth)
        human_mask = read_human_mask(self.mask_paths[index], kpts_2d)

        # 2d kpts
        kpts_2d = off_set_kpts(kpts=kpts_2d, leftTop=[0, 0])
        kpts_2d = scale_kpts(kpts=kpts_2d, height_ratio=1.0 * self.leres_size / origin_image.size[0],
                             width_ratio=1.0 * self.leres_size / origin_image.size[1])

        # hmr
        scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        scale = 1.4
        top, left, height, weight = get_torch_image_cut_box(leftTop=box[0], rightBottom=box[1], ExpandsRatio=scale)
        hmr_image = torchvision.transforms.functional.crop(origin_image, top, left, height, weight)
        transl, shape, pose = self.transls[index], self.shapes[index], self.poses[index]
        theta = np.concatenate((transl, pose, shape), axis=0)

        return {
            'leres_image': self.leres_transforms(leres_image),
            'hmr_image': self.hmr_transforms(hmr_image),
            'depth': self.depth_transforms(depth),
            'kpts_2d': torch.from_numpy(kpts_2d).float(),
            'kpts_3d': torch.from_numpy(kpts_3d).float(),
            'theta': torch.from_numpy(theta).float(),
            'focal_length': torch.from_numpy(focal_length).float(),
            'human_mask': torch.from_numpy(human_mask).bool(),
            'intrinsic':torch.from_numpy(intrinsic).float()
        }


if __name__ == '__main__':
    data_dir = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48-add-transl'
    gta_dataset = GTA_Dataset(data_dir)
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
        kpts_2d=i['kpts_2d']

        import matplotlib.pyplot as plt
        import numpy as np

        f, axarr = plt.subplots(1, 2)
        plt.figure()
        leres_image=leres_image.permute(1, 2, 0)
        hmr_image=hmr_image.permute(1,2,0)
        axarr[0].imshow(leres_image)
        axarr[1].imshow(hmr_image)
        # plt.imshow(hmr_image.permute(1, 2, 0))
        for j in range(24):
            # plt.imshow(img)
            axarr[0].scatter(np.squeeze(kpts_2d)[j][0],np.squeeze(kpts_2d)[j][1], s=50, c='red', marker='o')
        plt.show()
        # break
