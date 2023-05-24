from d_loss.align_loss_new_debug import AlignLoss
from datasets.gta_im_all_10801920_plane_change_intri_only_rotate import GTADataset
from torch.utils.data import DataLoader
from hmr_leres_model_all_fast_debug_plane_origin_align_only_rotate import HMRLeReS
import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    data_dir = '/Users/lwz/torch_ds/gta-im-test/FPS-5'
    batchsize = 4
    gta_dataset = GTADataset(data_dir)
    gta_loader = DataLoader(gta_dataset, batch_size=batchsize, shuffle=False)
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
        # pred_kpts_2d, pred_kpts_3d, verts = model_b.get_smpl_kpts_verts(transl=gt_smpl_transl,
        #                                                                 pose=gt_smpl_poses,
        #                                                                 shape=gt_smpl_shapes,
        #                                                                 focal_length=gt_focal_length / 5)
        pred_kpts_2d, pred_kpts_3d, verts = model_b.get_smpl_kpts_verts(transl=gt_smpl_transl,
                                                                        pose=gt_smpl_poses,
                                                                        shape=gt_smpl_shapes,
                                                                        focal_length=gt_focal_length)
        # faces = model_b.smpl_model.faces
        faces = torch.tensor([model_b.smpl_model.faces], device=model_b.device)
        mesh_mask_gt, mesh_edge_dist_gt, mesh_edge_dist_gt2 = model_a.vis_forward(faces, verts, verts, depth,
                                                                                  batch)
        # _ = model_a.forward(faces, verts, verts, depth, batch)

        for i in range(batchsize):
            leres_image = batch['leres_image'][i]
            hmr_image = batch['hmr_image'][i]
            kpts_2d = pred_kpts_2d[i]
            mesh_project_image = leres_image
            mesh_project_image[~(mesh_mask_gt[i].unsqueeze(0).repeat(3, 1, 1))] *= 0
            mesh_edge_dist_image = mesh_edge_dist_gt[i]
            mesh_edge_dist_image2 = mesh_edge_dist_gt2[i]
            f, ax = plt.subplots(2, 2)
            plt.figure()
            leres_image = leres_image.permute(1, 2, 0)
            hmr_image = hmr_image.permute(1, 2, 0)
            ax[0][0].imshow(leres_image)
            ax[0][0].set_title('LeReS Image')
            ax[0][1].imshow(hmr_image)
            ax[0][1].set_title('HMR Image')
            # ax[2].imshow(mesh_project_image)
            # ax[2].set_title('Mesh Project Image')
            ax[1][0].imshow(mesh_edge_dist_image[0], cmap='jet')
            ax[1][0].set_title('Manhattan Distance Transform')
            ax[1][1].imshow(mesh_edge_dist_image2[0], cmap='jet')
            ax[1][1].set_title('Euclidean Distance Transform')
            for j in range(24):
                ax[0][0].scatter(np.squeeze(kpts_2d)[j][0], np.squeeze(kpts_2d)[j][1], s=50, c='red', marker='o')
            plt.show()

        if idx > 4:
            break
