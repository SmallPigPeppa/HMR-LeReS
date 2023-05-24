import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import pickle

class MeshDataset(Dataset):
    def __init__(self, data_dir, use_flip=True, flip_prob=0.3):
        super().__init__()
        self.data_dir = data_dir
        self.scene_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if
                           os.path.isdir(os.path.join(data_dir, d))]
        self.use_flip = use_flip
        self.flip_prob = flip_prob

        self.load_data_set()

    def load_data_set(self):
        print('start loading mosh data.')
        self.num_samples = 0
        self.transls = []
        self.shapes = []
        self.poses = []
        for scene_dir in self.scene_dirs:
            info_smpl = np.load(os.path.join(scene_dir, 'info_smpl.pkl'), allow_pickle=True)
            total_shape = np.array(info_smpl['betas'])
            total_global_pose = np.array(info_smpl['global_orient'])
            total_body_pose = np.array(info_smpl['body_pose'])
            total_pose = np.concatenate((total_global_pose, total_body_pose), axis=1)
            total_transl = np.array(info_smpl['transl'])
            self.num_samples += len(total_pose)
            scence_samples = len(total_pose)
            for idx in range(scence_samples):
                self.shapes.append(total_shape[0].copy())
                self.poses.append(total_pose[idx].copy())
                self.transls.append(total_transl[idx].copy())

        print(f'finished load mosh data from {self.data_dir}, total {self.num_samples} samples')


    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        trival, pose, shape = self.transls[index], self.poses[index], self.shapes[index]
        if self.use_flip and random.uniform(0, 1) <= self.flip_prob:  # left-right reflect the pose
            # todo: pose = reflect_pose(pose)
            pass

        return {
            'theta': torch.from_numpy(np.concatenate((trival, pose, shape), axis=0)).float()
        }
