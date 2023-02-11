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
from torch.utils.data import Dataset, DataLoader
from hmr_leres_config import args
from hmr_data_utils import get_torch_image_cut_box, collect_valid_kpts, calc_aabb, off_set_kpts, scale_kpts,off_set_scale_kpts
from leres_data_utils import read_depthmap, read_human_mask, pil_loader
from PIL import Image


class GTA_Dataset(Dataset):
    def __init__(self, data_dir, scale_range=[1.2, 1.4], use_flip=False, flip_prob=0.3):
        self.data_dir = data_dir
        self.hmr_scale_range = [1.3,1.5]
        self.leres_scale_range = [2.5,5.0]
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
        self.joints_2d = []
        self.joints_3d = []
        self.transls = []
        self.shapes = []
        self.poses = []
        self.cam_focal_length = []
        self.cam_near_clips = []
        self.cam_far_clips = []

        total_kpts2d = np.array(self.info_hmr['kpts_2d'])
        total_kpts3d = np.array(self.info_hmr['kpts_3d'])
        total_joints2d = np.array(self.info_hmr['joints_2d'])
        total_joints3d = np.array(self.info_hmr['joints_3d'])
        total_shape = np.array(self.info_hmr['shape'])
        total_pose = np.array(self.info_hmr['pose'])
        total_transl = np.array(self.info_hmr['transl'])

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
        box = self.boxs[index]
        kpts_2d = self.kpts_2d[index]
        kpts_3d = self.kpts_3d[index]
        joints_2d=self.joints_2d[index]
        joints_3d=self.joints_3d[index]

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


        # crop and rescale leres_image,depth,human_mask,kpts2d,joints2d
        leres_scale = np.random.rand(4) * (self.leres_scale_range[1] - self.leres_scale_range[0]) + self.leres_scale_range[0]
        top, left, height, width = get_torch_image_cut_box(leftTop=box[0], rightBottom=box[1], ExpandsRatio=leres_scale)
        leres_cut_box=np.array([top, left, height, width])
        leres_image = torchvision.transforms.functional.crop(origin_image, top, left, height, width)
        depth=torchvision.transforms.functional.crop(depth, top, left, height, width)
        human_mask=human_mask# todo: crop




        kpts_2d=off_set_scale_kpts(kpts_2d,left=left,top=top,height_ratio=self.leres_size/height,width_ratio=self.leres_size/width)
        joints_2d = off_set_scale_kpts(joints_2d, left=left, top=top, height_ratio=self.leres_size / height,
                                     width_ratio=self.leres_size / width)

        # # 2d kpts
        # kpts_2d = off_set_kpts(kpts=kpts_2d, leftTop=[0, 0])
        # joints_2d=off_set_kpts(kpts=joints_2d,leftTop=[0, 0])
        # # kpts_2d = scale_kpts(kpts=kpts_2d, height_ratio=1.0 * self.leres_size / origin_image.size[0],
        # #                      width_ratio=1.0 * self.leres_size / origin_image.size[1])

        # hmr
        hmr_scale = np.random.rand(4) * (self.hmr_scale_range[1] - self.hmr_scale_range[0]) + self.hmr_scale_range[0]
        # scale = 1.4
        hmr_top, hmr_left, hmr_height, hmr_width = get_torch_image_cut_box(leftTop=box[0], rightBottom=box[1], ExpandsRatio=hmr_scale)
        hmr_image = torchvision.transforms.functional.crop(origin_image, hmr_top, hmr_left, hmr_height, hmr_width)
        transl, shape, pose = self.transls[index], self.shapes[index], self.poses[index]
        theta = np.concatenate((transl, pose, shape), axis=0)

        boody_pose = pose[3:]
        global_pose = pose[:3]

        return {
            'leres_image': self.leres_transforms(leres_image),
            'hmr_image': self.hmr_transforms(hmr_image),
            'depth': self.depth_transforms(depth),
            'kpts_2d': torch.from_numpy(kpts_2d).float(),
            'kpts_3d': torch.from_numpy(kpts_3d).float(),
            'joints_2d': torch.from_numpy(joints_2d).float(),
            'joints_3d': torch.from_numpy(joints_3d).float(),
            'theta': torch.from_numpy(theta).float(),
            'body_pose': torch.from_numpy(boody_pose).float(),
            'global_pose': torch.from_numpy(global_pose).float(),
            'focal_length': torch.from_numpy(focal_length).float(),
            'human_mask': torch.from_numpy(human_mask).bool(),
            'intrinsic': torch.from_numpy(intrinsic).float(),
            'leres_cut_box':torch.from_numpy(leres_cut_box)
        }


if __name__ == '__main__':
    data_dir = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48-add-transl'
    gta_dataset = GTA_Dataset(data_dir)
    gta_loader = DataLoader(
        dataset=gta_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )
    batch = next(iter(gta_loader))
    from d_loss.align_loss import AlignLoss
    loss_align = AlignLoss(K_intrin=batch['intrinsic'], image_size=[[1080, 1920]])
    smpl_model = '../HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt'
    smpl_model_dir = '../HMR/HMR-data/smpl'
    from SMPL import SMPL

    smpl = SMPL(smpl_model, obj_saveable=True)
    transl = batch['theta'][:, :3].contiguous()
    pose = batch['theta'][:, 3:75].contiguous()
    shape = batch['theta'][:, 75:].contiguous()
    verts, kpts_3d, Rs = smpl.forward(shape, pose, get_skin=True)

    kpts_3d += transl.unsqueeze(dim=1)
    verts += transl.unsqueeze(dim=1)
    # smpl.save_obj(verts=verts[0], obj_mesh_name='debug.obj')

    verts = torch.unsqueeze(verts[0], 0)
    faces = torch.Tensor([smpl.faces])
    vismask, nearest_depth_map, farrest_depth_map = loss_align.get_depth_map_pytorch3d(verts, faces)

    leres_image = batch['leres_image'][0]



    #
    # leres_cut_box=batch['leres_cut_box']
    [top, left, height, width]=batch['leres_cut_box'][0]
    # vismask=vismask.to(torch.long)
    vismask=torchvision.transforms.functional.crop(vismask,top, left, height, width)

    vismask=torchvision.transforms.functional.resize(vismask,size=[448,448],interpolation = InterpolationMode.NEAREST)


    leres_image[~(vismask.repeat(3, 1, 1))] *= 0
    hmr_image = batch['hmr_image'][0]
    kpts_2d = batch['kpts_2d'][0]
    kpts_2d = batch['joints_2d'][0]

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



