from util import align_by_pelvis, batch_rodrigues, copy_state_dict
import torch
from model import HMRNetBase
from merge_gta_dataset import MergeGTADataset
import os
import numpy as np
from Discriminator import Discriminator


from merge_config import args
from hmr_model_pl import HMR

data_set_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
pix_format = 'NCHW'
normalize = True
flip_prob = 0.5
use_flip = False
mdataset = MergeGTADataset(opt=args,
                           dataset_name='2020-06-11-10-06-48',
                           data_set_path=data_set_path,
                           use_crop=True,
                           scale_range=[1.1, 2.0],
                           use_flip=True,
                           min_pts_required=5,
                           pix_format=pix_format,
                           normalize=normalize,
                           flip_prob=flip_prob, )
# print('mdataset.leres_dataset[0]',mdataset.leres_dataset[0])
# print('mdataset.hmr_dataset[0]',mdataset.hmr_dataset[0])
print('mdataset[559]', mdataset[559])
print('len(mdataset)', len(mdataset))
images=torch.unsqueeze(mdataset[559]['image'], 0).cuda()
generator_dir= 'HMR/HMR-data/out-model-old'
generator_path = ''
# for ckpt in os.listdir(generator_dir):
#     if 'generator' in ckpt:
#         generator_path=os.path.join(generator_dir,ckpt)


for ckpt in os.listdir(generator_dir):
    if 'generator' in ckpt and '1500_20' in ckpt:
        generator_path=os.path.join(generator_dir,ckpt)



generator = HMRNetBase().cuda()
copy_state_dict(
    generator.state_dict(),
    torch.load(generator_path),
    prefix='module.'
)
generator.eval()
generator_outputs = generator(images)


def _accumulate_thetas(generator_outputs):
    thetas = []
    for (theta, verts, j2d, j3d, Rs) in generator_outputs:
        thetas.append(theta)
    return torch.cat(thetas, 0)

total_predict_thetas = _accumulate_thetas(generator_outputs)


(predict_theta, predict_verts, predict_j2d, predict_j3d, predict_Rs) = generator_outputs[-1]
pose = predict_theta[:, 3:75]
shape = predict_theta[:, 75:]
mesh=np.squeeze(predict_verts.cpu().detach().numpy())
generator.smpl.save_obj(verts=mesh,obj_mesh_name='demo_mesh.obj')
print('finished')