import sys
import numpy as np
import cv2
import os
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from datasets.hmr_data_utils import get_torch_image_cut_box, get_torch_image_cut_box_new, collect_valid_kpts, calc_aabb, \
    off_set_scale_kpts
from datasets.leres_data_utils import read_depthmap, read_human_mask, pil_loader, read_plane_mask
from PIL import Image


class GTADataset(Dataset):
    def __init__(self, data_dir, flip_prob=0.):
        super().__init__()
        self.data_dir = data_dir
        # import pdb;pdb.set_trace()
        self.scene_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if
                           os.path.isdir(os.path.join(data_dir, d))]
        self.hmr_scale_range = [1.3, 1.5]
        self.leres_scale_range = [1.5, 3.0]
        self.leres_aspect_ratio_range = [0.6, 1.7]
        # self.leres_aspect_ratio_range=[1.0,1.0]
        self.leres_area_ratio_range = [4.0, 25.0]
        self.flip_prob = flip_prob
        self.hmr_size = 224
        self.scale = 5
        self.leres_size = [1080 // self.scale, 1920 // self.scale]
        # self.fix_focal_length = 231.6068
        self.hmr_transforms = T.Compose([T.Resize((self.hmr_size, self.hmr_size)), T.ToTensor()])
        self.leres_transforms = T.Compose([T.Resize((self.leres_size[0], self.leres_size[1])), T.ToTensor()])
        self.depth_transforms = T.Compose(
            [T.Resize((self.leres_size[0], self.leres_size[1]), interpolation=InterpolationMode.NEAREST),
             T.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)))])
        self.mask_transforms = T.Compose(
            [T.Resize((self.leres_size[0], self.leres_size[1]), interpolation=InterpolationMode.NEAREST),
             T.Lambda(lambda image: torch.from_numpy(np.array(image).astype(bool)))])
        self.plane_mask_transforms = T.Compose(
            [T.Resize((self.leres_size[0], self.leres_size[1]), interpolation=InterpolationMode.NEAREST),
             T.Lambda(lambda image: torch.from_numpy(np.array(image).astype(int)))])
        self.load_data_set()

    def load_data_set(self):
        print('start loading gta-im data.')
        self.num_samples = 0
        self.image_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.plane_mask_paths = []
        self.boxs = []
        self.kpts_2d = []
        self.kpts_3d = []
        self.transls = []
        self.shapes = []
        self.poses = []
        self.cam_focal_length = []
        self.cam_near_clips = []
        self.cam_far_clips = []
        self.gta_heads_2d = []
        self.intrinsics = []

        for scene_dir in self.scene_dirs:
            info_npz = np.load(os.path.join(scene_dir, 'info_frames.npz'))
            info_pkl = np.load(os.path.join(scene_dir, 'info_frames.pickle'), allow_pickle=True)
            info_smpl = np.load(os.path.join(scene_dir, 'info_smpl.pkl'), allow_pickle=True)
            # info_smpl = np.load(os.path.join(scene_dir, '001_all.pkl'), allow_pickle=True)
            total_kpts3d = np.array(info_smpl['keypoints_3d'])
            total_shape = np.array(info_smpl['betas'])
            total_global_pose = np.array(info_smpl['global_orient'])
            total_body_pose = np.array(info_smpl['body_pose'])
            total_pose = np.concatenate((total_global_pose, total_body_pose), axis=1)
            total_transl = np.array(info_smpl['transl'])
            self.num_samples += len(total_pose)
            scence_samples = len(total_pose)
            rvec = np.zeros([3, 1], dtype=float)
            tvec = np.zeros([3, 1], dtype=float)
            dist_coeffs = np.zeros([4, 1], dtype=float)
            for idx in range(scence_samples):
                intrinsic_i = info_npz['intrinsics'][idx]
                # intrinsic_i_new = intrinsic_i / self.scale
                # intrinsic_i_new[2, 2] = 1  # Restore the [2, 2] element back to 1
                # x = total_kpts3d[idx]
                kpts2d_i, _ = cv2.projectPoints(total_kpts3d[idx], rvec, tvec, intrinsic_i, dist_coeffs)
                kpts2d_i = np.squeeze(kpts2d_i)
                lt, rb = calc_aabb(kpts2d_i)
                self.boxs.append((lt, rb))  # left-top, right-bottom
                self.kpts_2d.append(kpts2d_i.copy())
                self.kpts_3d.append(total_kpts3d[idx].copy())
                self.shapes.append(total_shape[0].copy())
                self.poses.append(total_pose[idx].copy())
                self.transls.append(total_transl[idx].copy())
                self.gta_heads_2d.append(kpts2d_i[15, :].copy())
                self.intrinsics.append(intrinsic_i.copy())

                img_path_i = os.path.join(scene_dir, '{:05d}'.format(idx) + '.jpg')
                mask_path_i = os.path.join(scene_dir, '{:05d}'.format(idx) + '_id.png')
                plane_mask_path_i = os.path.join(scene_dir, '{:05d}'.format(idx) + '_plane.png')
                depth_path_i = os.path.join(scene_dir, '{:05d}'.format(idx) + '.png')
                assert os.path.exists(img_path_i) and os.path.exists(mask_path_i) and os.path.exists(depth_path_i)
                self.image_paths.append(img_path_i)
                self.mask_paths.append(mask_path_i)
                self.plane_mask_paths.append(plane_mask_path_i)
                self.depth_paths.append(depth_path_i)

                info_i = info_pkl[idx]
                cam_near_clip_i = info_i['cam_near_clip']
                if 'cam_far_clip' in info_i.keys():
                    cam_far_clip_i = info_i['cam_far_clip']
                else:
                    cam_far_clip_i = 800.
                self.cam_near_clips.append(cam_near_clip_i)
                self.cam_far_clips.append(cam_far_clip_i)

            print(f'finished load from {scene_dir}, total {scence_samples} samples')
        self.fix_focal_length = self.intrinsics[0][0, 0] / self.scale
        print(f'finished load all gta-im data from {self.data_dir}, total {self.num_samples} samples')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # index=563
        # index=192
        # index=3
        index = 599

        origin_image = None
        depth = None
        human_mask = None
        while origin_image is None or depth is None or human_mask is None:
            image_path = self.image_paths[index]
            depth_path = self.depth_paths[index]
            mask_path = self.mask_paths[index]
            plane_mask_path = self.plane_mask_paths[index]
            origin_image = pil_loader(image_path)
            depth = read_depthmap(depth_path, self.cam_near_clips[index], self.cam_far_clips[index])
            human_mask = read_human_mask(mask_path, self.gta_heads_2d[index])
            plane_mask = read_plane_mask(plane_mask_path)
            if origin_image is None or depth is None or human_mask is None or plane_mask is None:
                print(f"index {index} wrong!")
                index = (index + 1) % len(self.image_paths)

        original_focal_length = self.intrinsics[index][0, 0]
        scaled_focal_length = original_focal_length / self.scale
        fix_focal_length = self.fix_focal_length
        # if scaled_focal_length != self.fix_focal_length:
        depth_ratio = fix_focal_length / scaled_focal_length
        depth *= depth_ratio

        # image_path = self.image_paths[index]
        # depth_path=self.depth_paths[index]
        # mask_path=self.mask_paths[index]
        # kpts_2d = self.kpts_2d[index][:24]
        # kpts_3d = self.kpts_3d[index][:24]
        # gta_head_2d = self.gta_heads_2d[index]
        # origin_image = pil_loader(image_path)
        # human_mask = read_human_mask(mask_path, gta_head_2d)
        # depth = read_depthmap(depth_path, self.cam_near_clips[index], self.cam_far_clips[index])

        # kpts
        box = self.boxs[index]
        kpts_2d = self.kpts_2d[index]
        kpts_3d = self.kpts_3d[index]

        # leres

        intrinsic = self.intrinsics[index]
        focal_length = np.array(intrinsic[0][0]).astype(np.float32)
        depth = Image.fromarray(depth)
        human_mask = Image.fromarray(human_mask)
        plane_mask = Image.fromarray(plane_mask)

        # crop and rescale leres_image,depth,human_mask,kpts2d,joints2d,human_mask
        aspect_ratio = np.random.uniform(self.leres_aspect_ratio_range[0], self.leres_aspect_ratio_range[1])
        area_ratio = np.random.uniform(self.leres_area_ratio_range[0], self.leres_area_ratio_range[1])
        top, left, height, width = get_torch_image_cut_box_new(left_top=box[0], right_bottom=box[1],
                                                               aspect_ratio=aspect_ratio, area_ratio=area_ratio)
        top, left, height, width = 0, 0, 1080, 1920

        leres_cut_box = np.array([top, left, height, width])
        leres_image = T.functional.crop(origin_image, top=top, left=left, height=height, width=width)
        depth = T.functional.crop(depth, top=top, left=left, height=height, width=width)
        human_mask = T.functional.crop(human_mask, top=top, left=left, height=height, width=width)
        plane_mask = T.functional.crop(plane_mask, top=top, left=left, height=height, width=width)
        kpts_2d_origin = kpts_2d.copy()
        kpts_2d = off_set_scale_kpts(kpts_2d, left=left, top=top, height_ratio=self.leres_size[0] / height,
                                     width_ratio=self.leres_size[1] / width)

        # hmr
        hmr_top, hmr_left, hmr_height, hmr_width = get_torch_image_cut_box_new(left_top=box[0], right_bottom=box[1])
        hmr_image = torchvision.transforms.functional.crop(origin_image, top=hmr_top, left=hmr_left, height=hmr_height,
                                                           width=hmr_width)

        transl, shape, pose = self.transls[index], self.shapes[index], self.poses[index]
        theta = np.concatenate((transl, pose, shape), axis=0)

        boody_pose = pose[3:]
        global_pose = pose[:3]

        vis = np.ones((kpts_2d.shape[0], 1))
        kpts_2d = np.concatenate((kpts_2d, vis), axis=1)
        kpts_3d = np.concatenate((kpts_3d, vis), axis=1)
        kpts_2d_origin = np.concatenate((kpts_2d_origin, vis), axis=1)
        # import pdb;pdb.set_trace()

        intrinsic = self.intrinsics[index]
        intrinsic_scaled = intrinsic / self.scale
        intrinsic_scaled[2, 2] = 1  # Restore the [2, 2] element back to 1
        origin_focal_length = intrinsic_scaled[0][0]
        intrinsic_scaled[0, 0] = fix_focal_length
        intrinsic_scaled[1, 1] = fix_focal_length
        focal_length = np.array(intrinsic_scaled[0][0]).astype(np.float32)
        scaled_focal_length = np.array(scaled_focal_length).astype(np.float32)
        leres_cut_box = leres_cut_box / self.scale
        kpts_3d[:, 2] *= depth_ratio

        return {
            'image_path': image_path,
            'index': index,
            'leres_image': self.leres_transforms(leres_image),
            'hmr_image': self.hmr_transforms(hmr_image),
            'depth': self.depth_transforms(depth),
            'human_mask': self.mask_transforms(human_mask),
            'plane_mask': self.plane_mask_transforms(plane_mask),
            'kpts_2d': torch.from_numpy(kpts_2d).float(),
            'kpts_2d_origin': torch.from_numpy(kpts_2d_origin).float(),
            'kpts_3d': torch.from_numpy(kpts_3d).float(),
            'theta': torch.from_numpy(theta).float(),
            'body_pose': torch.from_numpy(boody_pose).float(),
            'global_pose': torch.from_numpy(global_pose).float(),
            'focal_length': torch.from_numpy(focal_length).float(),
            'scaled_focal_length': torch.from_numpy(scaled_focal_length).float(),
            # 'intrinsic': torch.from_numpy(intrinsic).float(),
            'intrinsic': torch.from_numpy(intrinsic_scaled).float(),
            'leres_cut_box': torch.from_numpy(leres_cut_box)
        }


if __name__ == '__main__':
    import os
    from tqdm import tqdm

    # 创建debug-out文件夹
    debug_out_dir = 'debug-out-fixfocal4'
    os.makedirs(debug_out_dir, exist_ok=True)

    from torch.utils.data import DataLoader

    data_dir = '/Users/lwz/torch_ds/gta-im-fixbug/FPS-5'
    # data_dir = '/share/wenzhuoliu/torch_ds/gta-im/FPS-5-test'
    gta_dataset = GTADataset(data_dir)
    gta_loader = DataLoader(gta_dataset, batch_size=1, shuffle=False)

    import matplotlib.pyplot as plt
    import numpy as np

    for i, batch in tqdm(enumerate(gta_loader)):
        leres_image = batch['leres_image']
        hmr_image = batch['hmr_image']
        kpts_2d = batch['kpts_2d']
        depth = batch['depth']
        fix_fl = batch['focal_length']
        scaled_fl = batch['scaled_focal_length']
        intrinsic = batch['intrinsic']
        kpts_3d = batch['kpts_3d']
        # if not torch.all(torch.eq(fix_fl, scaled_fl)):
        for j in range(leres_image.shape[0]):

            rvec = np.zeros([3, 1], dtype=float)
            tvec = np.zeros([3, 1], dtype=float)
            dist_coeffs = np.zeros([4, 1], dtype=float)
            intrinsic_j = intrinsic[j].cpu().detach().numpy()
            kpts_3d_j = kpts_3d[j][:, :3].cpu().detach().numpy().astype(np.float32)
            # intrinsic_i_new = intrinsic_i / self.scale
            # intrinsic_i_new[2, 2] = 1  # Restore the [2, 2] element back to 1
            # kpts_3d_j = np.random.random_sample((10, 3)).astype(np.float32)
            kpts2d_j, _ = cv2.projectPoints(kpts_3d_j, rvec, tvec, intrinsic_j, dist_coeffs)
            kpts2d_j = np.squeeze(kpts2d_j)
            kpts2d_j_true = kpts_2d[j]

            f, axarr = plt.subplots(1, 2)
            leres_image_j = leres_image[j].permute(1, 2, 0)

            hmr_image_j = hmr_image[j].permute(1, 2, 0)

            axarr[0].imshow(leres_image_j)
            axarr[0].set_title('Normalized Focal')

            axarr[1].imshow(leres_image_j)
            axarr[1].set_title('Original Image')
            for k in range(0, 24):
                # axarr[0].scatter(np.squeeze(kpts_2d[j])[k][0], np.squeeze(kpts_2d[j])[k][1], s=50, c='red',
                #                  marker='o')
                # if (0 <= np.squeeze(kpts2d_j)[k][0] < leres_image_j.shape[1]-15 and
                #     0 <= np.squeeze(kpts2d_j)[k][1] < leres_image_j.shape[0]-15):
                axarr[0].scatter(np.squeeze(kpts2d_j)[k][0], np.squeeze(kpts2d_j)[k][1], s=30, c='red',
                                 marker='o')
                # if (0 <= np.squeeze(kpts2d_j_true)[k][0] < leres_image_j.shape[1]-15 and
                #     0 <= np.squeeze(kpts2d_j_true)[k][1] < leres_image_j.shape[0]-15):
                axarr[1].scatter(np.squeeze(kpts2d_j_true)[k][0], np.squeeze(kpts2d_j_true)[k][1], s=30, c='red',
                                 marker='o')
            plt.show()
        if i == 0:
            break
            # plt.savefig(f'{debug_out_dir}/image_{i * 16 + j}.png')
            # plt.close()
        # print(batch['leres_cut_box'][0])
#
