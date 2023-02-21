import torch
from datasets.gta_im import GTADataset
from torch.utils.data import DataLoader
import numpy as np
from hmr_leres_config import args
from hmr_leres_model_new_new import HMRLeReS

if __name__=='__main__':

    data_dir = 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48-add-transl'
    gta_dataset = GTADataset(data_dir)
    gta_loader = DataLoader(
        dataset=gta_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )
    batch = next(iter(gta_loader))
    smpl_model_path = 'HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt'
    from a_models.smpl import SMPL
    smpl = SMPL(smpl_model_path, obj_saveable=True)

    ckpt_path= 'hmr-leres-ckpt/last-v6.ckpt'
    model=HMRLeReS.load_from_checkpoint(ckpt_path, strict=False)


    transl = batch['theta'][:, :3].contiguous()
    pose = batch['theta'][:, 3:75].contiguous()
    shape = batch['theta'][:, 75:].contiguous()
    verts, kpts_3d, Rs = model.smpl_model.forward(shape, pose, get_skin=True)

    kpts_3d += transl.unsqueeze(dim=1)
    verts += transl.unsqueeze(dim=1)
    model.smpl_model.save_obj(verts=verts[0], obj_mesh_name='gt_mesh.obj')



    hmr_images=batch['hmr_image']
    predict_smpl_thetas = model.hmr_generator(hmr_images)[-1]
    predict_smpl_transl = predict_smpl_thetas[:, :3].contiguous()
    predict_smpl_poses = predict_smpl_thetas[:, 3:75].contiguous()
    predict_smpl_shapes = predict_smpl_thetas[:, 75:].contiguous()

    verts, kpts_3d, Rs = model.smpl_model.forward(predict_smpl_shapes, predict_smpl_poses, get_skin=True)

    kpts_3d += transl.unsqueeze(dim=1)
    verts += transl.unsqueeze(dim=1)
    model.smpl_model.save_obj(verts=verts[0], obj_mesh_name='debug_mesh.obj')
    print('finished')