import torch
from merge_gta_dataset import MergeGTADataset
import numpy as np
from merge_config import args
from hmr_model_pl import HMR

if __name__=='__main__':
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
    print('mdataset[559]', mdataset[559])
    print('len(mdataset)', len(mdataset))
    images=torch.unsqueeze(mdataset[559]['image'], 0)


    ckpt_path= 'hmr-ckpt-backup/last.ckpt'
    hmr_model=HMR.load_from_checkpoint(ckpt_path)
    generator_outputs = hmr_model.hmr_generator(images)


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
    hmr_model.hmr_generator.smpl.save_obj(verts=mesh,obj_mesh_name='demo_mesh.obj')
    print('finished')