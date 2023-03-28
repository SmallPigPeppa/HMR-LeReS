import torch


def compute_mpjpe(pred, gt):
    diff = pred - gt
    dist = torch.norm(diff, dim=2)
    return torch.mean(dist)


def compute_pck(pred, gt, threshold):
    diff = pred - gt
    dist = torch.norm(diff, dim=2)
    correct = (dist < threshold).sum().float()
    return 100.0 * correct / dist.numel()


def compute_pve(pred, gt):
    diff = pred - gt
    dist = torch.norm(diff, dim=2)
    return torch.mean(dist)


def val_kpts_verts(pred_kpts_3d, gt_kpts_3d, pred_verts, gt_verts, pck_threshold):
    # Remove the probability of kpts
    gt_kpts_3d = gt_kpts_3d[:, :, :3]

    # align verts
    pred_pelvis = pred_kpts_3d[:, [0], :]
    gt_pelvis = gt_kpts_3d[:, [0], :]
    pred_verts_aligned = pred_verts - pred_pelvis
    gt_verts_aligned = gt_verts - gt_pelvis

    # compute metrics
    mpjpe = compute_mpjpe(pred_kpts_3d, gt_kpts_3d)
    pck = compute_pck(pred_kpts_3d, gt_kpts_3d, pck_threshold)
    pve = compute_pve(pred_verts_aligned, gt_verts_aligned)

    metrics_dict = {'metric_mpjpe': mpjpe, 'metric_pck': pck, 'metric_pve': pve}

    return metrics_dict


if __name__ == '__main__':
    pred_kpts = torch.rand([8, 23, 3])
    gt_kpts = torch.rand([8, 23, 4])
    gt_kpts = gt_kpts[:, :, :3]
    a = compute_mpjpe(pred_kpts, gt_kpts)
    b = compute_pck(pred_kpts, gt_kpts, threshold=0.2)
    print(a)
    # pred_vertices - pred_pelvis
