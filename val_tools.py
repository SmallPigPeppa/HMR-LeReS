import torch
import numpy as np
from pare.utils.eval_utils import reconstruction_error, compute_error_verts

def select_index(gt_depth, select_size=10000):
    valid_size = np.sum(gt_depth>0)
    try:
        p = np.random.choice(valid_size, select_size*2, replace=False)
    except:
        p = np.random.choice(valid_size, select_size*2*2, replace=True)
    np.random.shuffle(p)
    p1 = p[0:select_size*2:2]
    p2 = p[1:select_size*2:2]

    p12_index = {'p1': p1, 'p2': p2}
    return p12_index

def weighted_human_disagreement_rate(gt, pred):
    p12_index = select_index(gt)
    gt_reshape = np.reshape(gt, gt.size)
    pred_reshape = np.reshape(pred, pred.size)
    mask = gt > 0
    gt_p1 = gt_reshape[mask][p12_index['p1']]
    gt_p2 = gt_reshape[mask][p12_index['p2']]
    pred_p1 = pred_reshape[mask][p12_index['p1']]
    pred_p2 = pred_reshape[mask][p12_index['p2']]

    p12_rank_gt = np.zeros_like(gt_p1)
    p12_rank_gt[gt_p1 > gt_p2] = 1
    p12_rank_gt[gt_p1 < gt_p2] = -1

    p12_rank_pred = np.zeros_like(gt_p1)
    p12_rank_pred[pred_p1 > pred_p2] = 1
    p12_rank_pred[pred_p1 < pred_p2] = -1

    err = np.sum(p12_rank_gt != p12_rank_pred)
    valid_pixels = gt_p1.size
    return err, valid_pixels

def recover_metric_depth(pred, gt, mask0=None):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) #& (pred > 1e-8)
    if mask0 is not None and mask0.sum() > 0:
        if type(mask0).__module__ == torch.__name__:
            mask0 = mask0.cpu().numpy()
        mask0 = mask0.squeeze()
        mask0 = mask0 > 0
        mask = mask & mask0
    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)
    return pred_metric


def validate_rel_depth_err(pred, gt, smoothed_criteria, scale=10.):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    valid_mask = gt > 0
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    # invalid evaluation image
    if gt.size < 10:
        return smoothed_criteria

    # Scale matching
    #pred = recover_metric_depth(pred, gt)

    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    # Mean Absolute Relative Error
    rel = np.abs(gt_scale - pred_scale) / gt_scale  # compute errors
    abs_rel_sum = np.sum(rel)
    smoothed_criteria['err_absRel'].AddValue(np.float64(abs_rel_sum), n_pxl)

    # WHDR error
    whdr_err_sum, eval_num = weighted_human_disagreement_rate(gt_scale, pred_scale)
    smoothed_criteria['err_whdr'].AddValue(np.float64(whdr_err_sum), eval_num)
    return smoothed_criteria

def validation_step(batch):
    images = batch['img']
    imgnames = batch['imgname']
    dataset_names = batch['dataset_name']
    cam_rotmat = batch['cam_rotmat']
    cam_intrinsics = batch['cam_int']
    bbox_scale = batch['scale']
    bbox_center = batch['center']
    img_h = batch['orig_shape'][:, 0]
    img_w = batch['orig_shape'][:, 1]

    curr_batch_size = images.shape[0]

    with torch.no_grad():
        pred = self(images, cam_rotmat, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h)
        pred_vertices = pred['smpl_vertices']
        pred_joints_24 = self.smpl_native(
            shape=pred['pred_shape'],
            body_pose=pred['pred_pose'][:, 1:].contiguous(),
            global_orient=pred['pred_pose'][:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        ).joints[:, :24]

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

        gt_keypoints_3d = batch['pose_3d'].cuda()
        gt_joints_24 = batch['joints_24'].cuda()
        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        pred_pelvis_j_24 = pred_joints_24[:, [0], :].clone()
        pred_joints_24 = pred_joints_24 - pred_pelvis_j_24

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        # Reconstuction_error(PA-MPJPE)
        r_error, r_error_per_joint = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None,
        )

        error_j_24 = torch.sqrt(((pred_joints_24 - gt_joints_24) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        r_error_j_24, r_error_per_joint_j_24 = reconstruction_error(
            pred_joints_24.cpu().numpy(),
            gt_joints_24.cpu().numpy(),
            reduction=None,
        )

        # Per-vertex error
        if 'vertices' in batch.keys():
            gt_vertices = batch['vertices'].cuda()

            # logger.debug(f'GT vertices shape: {gt_vertices.shape}')
            # logger.debug(f'PR vertices shape: {pred_vertices.shape}')
            # logger.debug(f'ARRAY: {gt_vertices}')

            v2v = compute_error_verts(
                pred_verts=pred_vertices.cpu().numpy(),
                target_verts=gt_vertices.cpu().numpy(),
            )
            self.val_v2v += v2v.tolist()
        else:
            self.val_v2v += np.zeros_like(error).tolist()

        ####### DEBUG 3D JOINT PREDICTIONS and GT ###########
        # from ..utils.vis_utils import show_3d_pose
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(12, 7))
        # plt.title(f'error {error[0].item()*1000:.2f}, r_err {r_error[0].item()*1000:.2f}')
        # ax = fig.add_subplot('121', projection='3d', aspect='auto')
        # show_3d_pose(kp_3d=pred_joints_24[0].cpu(), ax=ax, dataset='smpl')
        #
        # ax = fig.add_subplot('122', projection='3d', aspect='auto')
        # show_3d_pose(kp_3d=gt_joints_24[0].cpu(), ax=ax, dataset='smpl')
        # plt.show()
        #####################################################

        self.val_mpjpe += error.tolist()
        self.val_pampjpe += r_error.tolist()
        self.val_mpjpe_24 += error_j_24.tolist()
        self.val_pampjpe_24 += r_error_j_24.tolist()

        error_per_joint = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
        error_per_joint_24 = torch.sqrt(((pred_joints_24 - gt_joints_24) ** 2).sum(dim=-1)).cpu().numpy()

        self.evaluation_results['mpjpe'] += error_per_joint[:, :14].tolist()
        self.evaluation_results['pampjpe'] += r_error_per_joint[:, :14].tolist()
        self.evaluation_results['mpjpe_24'] += error_per_joint_24.tolist()
        self.evaluation_results['pampjpe_24'] += r_error_per_joint_j_24.tolist()
        self.evaluation_results['imgname'] += imgnames
        self.evaluation_results['dataset_name'] += dataset_names

        if self.hparams.TESTING.SAVE_RESULTS:
            tolist = lambda x: [i for i in x.cpu().numpy()]
            self.evaluation_results['pose'] += tolist(pred['pred_pose'])
            self.evaluation_results['shape'] += tolist(pred['pred_shape'])
            self.evaluation_results['cam'] += tolist(pred['pred_cam'])
            self.evaluation_results['vertices'] += tolist(pred_vertices)

        if self.hparams.TESTING.SAVE_IMAGES and batch_nb % self.hparams.TESTING.SAVE_FREQ == 0:
            # this saves the rendered images
            self.validation_summaries(batch, pred, batch_nb, dataloader_nb)

    return {
        'mpjpe': error.mean(),
        'pampjpe': r_error.mean(),
        'per_mpjpe': error_per_joint,
        'per_pampjpe': r_error_per_joint
    }