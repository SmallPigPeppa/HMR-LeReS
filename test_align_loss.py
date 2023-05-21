from d_loss.align_loss_new_debug import AlignLoss
from datasets.gta_im_all_10801920_plane import GTADataset
from torch.utils.data import DataLoader
from hmr_leres_model_all_fast_debug_plane_origin import HMRLeReS
import torch

if __name__ == "__main__":
    data_dir = '/Users/lwz/torch_ds/gta-im-test/FPS-5'
    gta_dataset = GTADataset(data_dir)
    gta_loader = DataLoader(gta_dataset, batch_size=4, shuffle=False)
    model_a = AlignLoss()
    model_b = HMRLeReS()
    for idx, batch in enumerate(gta_loader):
        # transl = batch['transl']
        depth = batch['depth']
        human_mask = batch['human_mask']
        intrinsic = batch['intrinsic']
        theta = batch['theta']
        gt_focal_length = batch['focal_length']
        gt_smpl_shapes = theta[:, 75:].contiguous()
        gt_smpl_poses = theta[:, 3:75].contiguous()
        gt_smpl_transl = theta[:, :3].contiguous()
        pred_kpts_2d, pred_kpts_3d, verts = model_b.get_smpl_kpts_verts(transl=gt_smpl_transl,
                                                                        pose=gt_smpl_poses,
                                                                        shape=gt_smpl_shapes,
                                                                        focal_length=gt_focal_length / 5)
        # faces = model_b.smpl_model.faces
        faces = torch.tensor([model_b.smpl_model.faces], device=model_b.device)
        model_a.vis_batch_align_loss(verts, faces, depth, batch, pred_kpts_2d, pred_kpts_3d)
        if idx > 4:
            break
