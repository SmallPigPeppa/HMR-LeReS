import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


class MeshDataset(Dataset):
    def __init__(self, data_dir, use_flip=True, flip_prob=0.3):
        self.data_dir = data_dir
        self.use_flip = use_flip
        self.flip_prob = flip_prob

        self.load_data_set()

    def load_data_set(self):
        print('start loading mosh data.')
        info_hmr = np.load(os.path.join(self.data_dir, 'info_hmr.pickle'), allow_pickle=True)
        self.transl = np.array(info_hmr['transl'].copy())
        self.shapes = np.array(info_hmr['shape'].copy())
        self.poses = np.array(info_hmr['pose'].copy())
        print('finished load mosh data, total {} samples'.format(len(self.poses)))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        trival, pose, shape = self.transl[index], self.poses[index], self.shapes[index]

        if self.use_flip and random.uniform(0, 1) <= self.flip_prob:  # left-right reflect the pose
            # todo: pose = reflect_pose(pose)
            pass

        return {
            'theta': torch.from_numpy(np.concatenate((trival, pose, shape), axis=0)).float()
        }
