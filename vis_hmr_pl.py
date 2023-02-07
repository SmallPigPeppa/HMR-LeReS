import torch
from HMR.src.dataloader.gta_dataloader import gta_dataloader as hmr_dataset
import numpy as np
from hmr_leres_config import args
from hmr_model_pl import HMR

if __name__=='__main__':
    pix_format = 'NCHW'
    normalize = True
    flip_prob = 0.
    use_flip = False
    hmr_3d_path = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
    use_crop = True
    scale_range = [1.1, 2.0]
    min_pts_required = 5
    hmr_3d_dataset = hmr_dataset(hmr_3d_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,normalize, flip_prob)
    print('mdataset[559]',hmr_3d_dataset[559])
    print('len(mdataset)', len(hmr_3d_dataset))
    images=torch.unsqueeze(hmr_3d_dataset[559]['image'], 0)


    ckpt_path= 'hmr-ckpt-v6.0/last-v1.ckpt'
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
    hmr_model.hmr_generator.smpl.save_obj(verts=mesh,obj_mesh_name='debug_mesh.obj')
    print('finished')