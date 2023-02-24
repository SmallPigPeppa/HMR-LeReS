import torch
import numpy as np


def recover_metric_depth(pred, gt, ):
    a, b = np.polyfit(pred, gt, deg=1)
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred)
        gt_mean = np.mean(gt)
        pred_metric = pred * (gt_mean / pred_mean)
    return pred_metric


def val_depth(pred, gt, min_threshold=0, max_threshold=15):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().detach().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().detach().numpy()
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    valid_mask = (gt > min_threshold) & (gt < max_threshold) & (pred > min_threshold)
    gt = gt[valid_mask]
    pred = pred[valid_mask]


    # Scale matching
    pred = recover_metric_depth(pred, gt)

    n_pxl = gt.size

    # Mean Absolute Relative Error
    rel = np.abs(gt - pred)
    abs_rel_sum = np.sum(rel)

    # WHDR error
    whdr_err_sum, eval_num = weighted_human_disagreement_rate(gt, pred)

    metrics_dict = {'err_absRel': abs_rel_sum / n_pxl, 'err_whdr': whdr_err_sum / eval_num}

    return metrics_dict
    # criteria = {}
    # criteria['err_absRel'] = abs_rel_sum / n_pxl
    # criteria['err_whdr'] = whdr_err_sum / eval_num
    # return criteria


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


def select_index(gt_depth, select_size=10000):
    valid_size = np.sum(gt_depth > 0)
    try:
        p = np.random.choice(valid_size, select_size * 2, replace=False)
    except:
        p = np.random.choice(valid_size, select_size * 2 * 2, replace=True)
    np.random.shuffle(p)
    p1 = p[0:select_size * 2:2]
    p2 = p[1:select_size * 2:2]

    p12_index = {'p1': p1, 'p2': p2}
    return p12_index
