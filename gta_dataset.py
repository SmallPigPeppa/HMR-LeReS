'''
    file:   gta_dataloader.py

    author: wzliu
    date:   2022
    purpose:  load gta-im data
'''

import sys
from torch.utils.data import Dataset
import os
import numpy as np
import random
import cv2
import h5py
import torch

sys.path.append('./src')
from util import calc_aabb, cut_image, draw_lsp_14kp__bone, rectangle_intersect, \
    get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize, reflect_pose

from util_gta import flip_image_with_smpl_2dkp,reflect_smpl_3dkp
from hmr_leres_config import args
from timer import Clock


class gta_dataloader(Dataset):
    def __init__(self, data_set_path, use_crop, scale_range, use_flip, min_pts_required, pix_format='NHWC',
                 normalize=False, flip_prob=0.3):
        self.data_folder = data_set_path
        self.use_crop = use_crop
        self.scale_range = scale_range
        self.use_flip = use_flip
        self.flip_prob = flip_prob
        self.min_pts_required = min_pts_required
        self.pix_format = pix_format
        self.normalize = normalize
        self._load_data_set()

    def _load_data_set(self):

        clk = Clock()

        self.images = []
        self.kp2ds = []
        self.boxs = []
        self.kp3ds = []
        self.shapes = []
        self.poses = []
        self.masks=[]

        print('start loading gta-im data.')
        anno_file_path = os.path.join(self.data_folder, 'annot.h5')


        with h5py.File(anno_file_path) as fp:
            total_kp2d = np.array(fp['gt2d'])
            total_kp3d = np.array(fp['gt3d'])
            total_shape = np.array(fp['shape'])
            total_pose = np.array(fp['pose'])
            # print(total_pose.shape)

            assert len(total_kp2d) == len(total_kp3d) and \
                   len(total_kp2d) == len(total_shape) and len(total_kp2d) == len(total_pose)

            l = len(total_kp2d)

            def _collect_valid_pts(pts):
                r = []
                for pt in pts:
                    if pt[2] != 0:
                        r.append(pt)
                return r

            for index in range(l):
                kp2d = total_kp2d[index].reshape((-1, 3))
                # if np.sum(kp2d[:, 2]) < self.min_pts_required:
                #     continue

                lt, rb, v = calc_aabb(_collect_valid_pts(kp2d))
                self.kp2ds.append(np.array(kp2d.copy(), dtype=float))
                # left-top, right-bottom
                self.boxs.append((lt, rb))
                # self.kp3ds.append(total_kp3d[index].copy().reshape(-1, 3))
                self.kp3ds.append(total_kp3d[index].copy())
                self.shapes.append(total_shape[index].copy())
                self.poses.append(total_pose[index].copy())

                img_i = os.path.join(self.data_folder, '{:05d}'.format(index) + '.jpg')
                mask_i = os.path.join(self.data_folder, '{:05d}'.format(index) + '_id.png')
                assert os.path.exists(img_i) and os.path.exists(mask_i)
                self.images.append(img_i)
                self.masks.append(mask_i)
        print('finished load gta-im data, total {} samples'.format(len(self.kp3ds)))
        clk.stop()

    def __len__(self):
        return len(self.images)

    # def __getitem__(self, index):
    #     image_path = self.images[index]
    #     kps = self.kp2ds[index].copy()
    #     box = self.boxs[index]
    #     kp_3d = self.kp3ds[index].copy()
    #
    #
    #     scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
    #     image, kps = cut_image(image_path, kps, scale, box[0], box[1])
    #
    #     ratio = 1.0 * args.crop_size / image.shape[0]
    #     kps[:, :2] *= ratio
    #     dst_image = cv2.resize(image, (args.crop_size, args.crop_size), interpolation=cv2.INTER_CUBIC)
    #
    #     trival, shape, pose = np.zeros(3), self.shapes[index], self.poses[index]
    #
    #     if self.use_flip and random.random() <= self.flip_prob:
    #         dst_image, kps = flip_image_with_smpl_2dkp(dst_image, kps)
    #         pose = reflect_pose(pose)
    #         kp_3d = reflect_smpl_3dkp(kp_3d)
    #
    #     # normalize kp to [-1, 1]
    #     ratio = 1.0 / args.crop_size
    #     kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0
    #
    #     theta = np.concatenate((trival, pose, shape), axis=0)
    #
    #     return {
    #         'image': torch.from_numpy(
    #             convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
    #         'kp_2d': torch.from_numpy(kps).float(),
    #         'kp_3d': torch.from_numpy(kp_3d).float(),
    #         'theta': torch.from_numpy(theta).float(),
    #         'image_name': self.images[index],
    #         'w_smpl': 1.0,
    #         'w_3d': 1.0,
    #         'data_set': 'gta-im'
    #     }
    def __getitem__(self, index):
        # image_path = self.images[index]
        # kps = self.kp2ds[index].copy()
        # box = self.boxs[index]
        # kp_3d = self.kp3ds[index].copy()
        index=559

        image_path = self.images[index]
        kps = self.kp2ds[index]
        box = self.boxs[index]
        kp_3d = self.kp3ds[index]

        # scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        # scale = self.scale_range[1]
        scale = 1.0
        image, kps = cut_image(image_path, kps, scale, box[0], box[1])

        ratio = 1.0 * args.crop_size / image.shape[0]
        kps[:, :2] *= ratio
        dst_image = cv2.resize(image, (args.crop_size, args.crop_size), interpolation=cv2.INTER_CUBIC)

        trival, shape, pose = np.zeros(3), self.shapes[559], self.poses[559]

        if self.use_flip and random.random() <= self.flip_prob:
            dst_image, kps = flip_image_with_smpl_2dkp(dst_image, kps)
            pose = reflect_pose(pose)
            kp_3d = reflect_smpl_3dkp(kp_3d)

        # normalize kp to [-1, 1]
        ratio = 1.0 / args.crop_size
        kps[:, :2] = 2.0 * kps[:, :2] * ratio - 1.0

        theta = np.concatenate((trival, pose, shape), axis=0)

        return {
            'image': torch.from_numpy(
                convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            'kp_2d': torch.from_numpy(kps).float(),
            'kp_3d': torch.from_numpy(kp_3d).float(),
            'theta': torch.from_numpy(theta).float(),
            'image_name': self.images[index],
            'w_smpl': 1.0,
            'w_3d': 1.0,
            'data_set': 'gta-im'
        }


# if __name__ == '__main__':
# #     h36m = hum36m_dataloader('E:/HMR/data/human3.6m', True, [1.1, 2.0], True, 5, flip_prob=1)
#     data_dir='C:/Users/90532/Desktop/Datasets/HMR/2020-06-11-10-06-48'
#     gta_loader=gta_dataloader(data_dir, True, [1.1, 2.0], True, 5, flip_prob=1)
#     l = len(gta_loader)
#     for _ in range(l):
#         r = gta_loader.__getitem__(_)
#         print(r)
#         break

if __name__=='__main__':
    data_set_path='C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
    pix_format = 'NCHW'
    normalize = True
    flip_prob = 0.5
    use_flip = False
    gta_loader = gta_dataloader(
        data_set_path=data_set_path,
        use_crop=True,
        scale_range=[1.1, 2.0],
        use_flip=True,
        min_pts_required=5,
        pix_format=pix_format,
        normalize=normalize,
        flip_prob=flip_prob
    )
    # for idx,i in enumerate(iter(gta_loader)):
    #     if i['kp_2d'].shape!=i['kp_3d'].shape:
    #         print('frame{idx}'.format(idx=idx))
    #         print(i['kp_2d'].shape,i['kp_3d'].shape)
    for idx,i in enumerate(iter(gta_loader)):
        if len(i['kp_2d'])!=23:
            print('frame{idx}'.format(idx=idx))
            print(i['kp_2d'].shape,i['kp_3d'].shape)