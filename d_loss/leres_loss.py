import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from lib_train.models.Surface_normal import surface_normal_from_depth
from lib_train.models.PWN_edges import edgeGuidedSampling, randomSamplingNormal
from lib_train.models.ranking_loss import randomSampling,ind2sub,sub2ind


class DepthRegressionLoss(pl.LightningModule):
    """
    d_loss = MAE((d-u)/s - d') + MAE(tanh(0.01*(d-u)/s) - tanh(0.01*d'))
    """

    def __init__(self, min_threshold=-1e-8, max_threshold=1e8):
        super(DepthRegressionLoss, self).__init__()
        self.valid_threshold = min_threshold
        self.max_threshold = max_threshold
        # self.thres1 = 0.9

    def forward(self, pred_depth, gt_depth):
        # only l1
        """
        Calculate d_loss.
        """
        valid_mask = (gt_depth > self.valid_threshold) & (gt_depth < self.max_threshold)  # [b, c, h, w]
        valid_pred_depth = pred_depth[valid_mask]
        valid_gt_depth = gt_depth[valid_mask]
        l1_loss = F.l1_loss(valid_pred_depth, valid_gt_depth)

        tanh_valid_pred_depth = torch.tanh(0.01 * valid_pred_depth)
        tanh_valid_gt_depth = torch.tanh(0.01 * valid_gt_depth)
        tanh_l1_loss = F.l1_loss(tanh_valid_pred_depth, tanh_valid_gt_depth)

        # return l1_loss + tanh_l1_loss
        return l1_loss


class EdgeguidedNormalRegressionLoss(pl.LightningModule):
    def __init__(self, point_pairs=10000, cos_theta1=0.3, cos_theta2=0.95, cos_theta3=0.5, cos_theta4=0.86,
                 min_threshold=-1e-8, max_threshold=10.1):
        super(EdgeguidedNormalRegressionLoss, self).__init__()
        self.point_pairs = point_pairs  # number of point pairs
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.cos_theta1 = cos_theta1  # 75 degree
        self.cos_theta2 = cos_theta2  # 10 degree
        self.cos_theta3 = cos_theta3  # 60 degree
        self.cos_theta4 = cos_theta4  # 30 degree
        self.kernel = torch.tensor(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32), requires_grad=False)[
                      None, None, :, :]

    def getEdge(self, images):
        n, c, h, w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(self.device).view((1, 1, 3, 3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(self.device).view((1, 1, 3, 3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:, 0, :, :].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:, 0, :, :].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def getNormalEdge(self, normals):
        n, c, h, w = normals.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(self.device).view((1, 1, 3, 3)).repeat(3, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(self.device).view((1, 1, 3, 3)).repeat(3, 1, 1, 1)
        gradient_x = torch.abs(F.conv2d(normals, a, groups=c))
        gradient_y = torch.abs(F.conv2d(normals, b, groups=c))
        gradient_x = gradient_x.mean(dim=1, keepdim=True)
        gradient_y = gradient_y.mean(dim=1, keepdim=True)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def forward(self, pred_depths, gt_depths, images, focal_length):
        """
        inputs and targets: surface normal image
        images: rgb images
        """
        masks = (gt_depths > self.min_threshold) & (gt_depths < self.max_threshold)
        # pred_depths_ss = self.scale_shift_pred_depth(pred_depths, gt_depths)
        inputs = surface_normal_from_depth(pred_depths, focal_length, valid_mask=masks)
        targets = surface_normal_from_depth(gt_depths, focal_length, valid_mask=masks)
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)
        # find edges from normals
        edges_normal, thetas_normal = self.getNormalEdge(targets)
        mask_img_border = torch.ones_like(edges_normal)  # normals on the borders
        mask_img_border[:, :, 5:-5, 5:-5] = 0
        edges_normal[mask_img_border.bool()] = 0
        # find edges from depth
        edges_depth, _ = self.getEdge(gt_depths)
        edges_depth_mask = edges_depth.ge(edges_depth.max() * 0.1)
        edges_mask_dilate = torch.clamp(
            torch.nn.functional.conv2d(edges_depth_mask.float(), self.kernel.to(self.device), padding=(1, 1)), 0,
            1).bool()
        edges_normal[edges_mask_dilate] = 0
        edges_img[edges_mask_dilate] = 0
        # =============================
        n, c, h, w = targets.size()

        inputs = inputs.contiguous().view(n, c, -1).double()
        targets = targets.contiguous().view(n, c, -1).double()
        masks = masks.contiguous().view(n, -1)
        edges_img = edges_img.contiguous().view(n, -1).double()
        thetas_img = thetas_img.contiguous().view(n, -1).double()
        edges_normal = edges_normal.view(n, -1).double()
        thetas_normal = thetas_normal.view(n, -1).double()

        # initialization
        loss = torch.DoubleTensor([0.0]).to(self.device)

        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num, row_img, col_img = edgeGuidedSampling(
                inputs[i, :], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)
            normal_inputs_A, normal_inputs_B, normal_targets_A, normal_targets_B, normal_masks_A, normal_masks_B, normal_sample_num, row_normal, col_normal = edgeGuidedSampling(
                inputs[i, :], targets[i, :], edges_normal[i], thetas_normal[i], masks[i, :], h, w)

            # Combine EGS + EGNS
            inputs_A = torch.cat((inputs_A, normal_inputs_A), 1)
            inputs_B = torch.cat((inputs_B, normal_inputs_B), 1)
            targets_A = torch.cat((targets_A, normal_targets_A), 1)
            targets_B = torch.cat((targets_B, normal_targets_B), 1)
            masks_A = torch.cat((masks_A, normal_masks_A), 0)
            masks_B = torch.cat((masks_B, normal_masks_B), 0)

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A & masks_B

            # GT ordinal relationship
            target_cos = torch.abs(torch.sum(targets_A * targets_B, dim=0))
            input_cos = torch.abs(torch.sum(inputs_A * inputs_B, dim=0))
            # ranking regression
            # d_loss += torch.mean(torch.abs(target_cos[consistency_mask] - input_cos[consistency_mask]))

            # Ranking for samples
            mask_cos75 = target_cos < self.cos_theta1
            mask_cos10 = target_cos > self.cos_theta2
            # Regression for samples
            loss += torch.sum(
                torch.abs(target_cos[mask_cos75 & consistency_mask] - input_cos[mask_cos75 & consistency_mask])) / (
                            torch.sum(mask_cos75 & consistency_mask) + 1e-8)
            loss += torch.sum(
                torch.abs(target_cos[mask_cos10 & consistency_mask] - input_cos[mask_cos10 & consistency_mask])) / (
                            torch.sum(mask_cos10 & consistency_mask) + 1e-8)

            # Random Sampling regression
            random_sample_num = torch.sum(mask_cos10 & consistency_mask) + torch.sum(
                torch.sum(mask_cos75 & consistency_mask))
            random_inputs_A, random_inputs_B, random_targets_A, random_targets_B = randomSamplingNormal(inputs[i, :],
                                                                                                        targets[i, :],
                                                                                                        masks[i, :],
                                                                                                        random_sample_num)
            # GT ordinal relationship
            random_target_cos = torch.abs(torch.sum(random_targets_A * random_targets_B, dim=0))
            random_input_cos = torch.abs(torch.sum(random_inputs_A * random_inputs_B, dim=0))
            loss += torch.sum(torch.abs(random_target_cos - random_input_cos)) / (random_target_cos.shape[0] + 1e-8)

        if loss[0] != 0:
            return loss[0].float() / n
        else:
            return pred_depths.sum() * 0.0


def recover_scale_shift_depth(pred, gt, min_threshold=1e-8, max_threshold=1e8):
    b, c, h, w = pred.shape
    mask = (gt > min_threshold) & (gt < max_threshold)  # [b, c, h, w]
    EPS = 1e-6 * torch.eye(2, dtype=pred.dtype, device=pred.device)
    scale_shift_batch = []
    ones_img = torch.ones((1, h, w), dtype=pred.dtype, device=pred.device)
    for i in range(b):
        mask_i = mask[i, ...]
        pred_valid_i = pred[i, ...][mask_i]
        ones_i = ones_img[mask_i]
        pred_valid_ones_i = torch.stack((pred_valid_i, ones_i), dim=0)  # [c+1, n]
        A_i = torch.matmul(pred_valid_ones_i, pred_valid_ones_i.permute(1, 0))  # [2, 2]
        A_inverse = torch.inverse(A_i + EPS)

        gt_i = gt[i, ...][mask_i]
        B_i = torch.matmul(pred_valid_ones_i, gt_i)[:, None]  # [2, 1]
        scale_shift_i = torch.matmul(A_inverse, B_i)  # [2, 1]
        scale_shift_batch.append(scale_shift_i)
    scale_shift_batch = torch.stack(scale_shift_batch, dim=0)  # [b, 2, 1]
    ones = torch.ones_like(pred)
    pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
    pred_scale_shift = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2),
                                    scale_shift_batch)  # [b, h*w, 1]
    pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))
    return pred_scale_shift


class MultiScaleGradLoss(pl.LightningModule):
    """
    Our proposed GT normalized Multi-scale Gradient Loss Fuction.
    """

    def __init__(self, scale=4, min_threshold=-1e-8, max_threshold=1e8):
        super(MultiScaleGradLoss, self).__init__()
        self.scales_num = scale
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.EPSILON = 1e-6

    def one_scale_gradient_loss(self, pred_scale, gt, mask):
        mask_float = mask.to(dtype=pred_scale.dtype, device=self.device)

        d_diff = pred_scale - gt

        v_mask = torch.mul(mask_float[:, :, :-2, :], mask_float[:, :, 2:, :])
        v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
        h_mask = torch.mul(mask_float[:, :, :, :-2], mask_float[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        valid_num = torch.sum(h_mask) + torch.sum(v_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / (valid_num + 1e-8)

        return gradient_loss

    def transform(self, gt):
        # Get mean and standard deviation
        data_mean = []
        data_std_dev = []
        for i in range(gt.shape[0]):
            gt_i = gt[i]
            mask = gt_i > 0
            depth_valid = gt_i[mask]
            if depth_valid.shape[0] < 10:
                data_mean.append(torch.tensor(0).to(self.device))
                data_std_dev.append(torch.tensor(1).to(self.device))
                continue
            size = depth_valid.shape[0]
            depth_valid_sort, _ = torch.sort(depth_valid, 0)
            depth_valid_mask = depth_valid_sort[int(size * 0.1): -int(size * 0.1)]
            data_mean.append(depth_valid_mask.mean())
            data_std_dev.append(depth_valid_mask.std())
        data_mean = torch.stack(data_mean, dim=0).to(self.device)
        data_std_dev = torch.stack(data_std_dev, dim=0).to(self.device)

        return data_mean, data_std_dev

    def forward(self, pred, gt):
        mask = (gt > self.min_threshold) & (gt < self.max_threshold)
        grad_term = 0.0
        gt_mean, gt_std = self.transform(gt)
        gt_trans = gt
        for i in range(self.scales_num):
            step = pow(2, i)
            d_gt = gt_trans[:, :, ::step, ::step]
            d_pred = pred[:, :, ::step, ::step]
            d_mask = mask[:, :, ::step, ::step]
            grad_term += self.one_scale_gradient_loss(d_pred, d_gt, d_mask)
        return grad_term


class EdgeguidedRankingLoss(pl.LightningModule):
    def __init__(self, point_pairs=10000, sigma=0.03, alpha=1.0, min_threshold=-1e-8, max_threshold=15):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs  # number of point pairs
        self.sigma = sigma  # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha  # used for balancing the effect of = and (<,>)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        # self.regularization_loss = GradientLoss(scales=4)

    def getEdge(self, images):
        n, c, h, w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(self.device).view((1, 1, 3, 3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(self.device).view((1, 1, 3, 3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:, 0, :, :].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:, 0, :, :].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)

        return edges, thetas

    def forward(self, inputs, targets, images):

        masks = (targets > self.min_threshold) & (targets < self.max_threshold)
        # Comment this line if you don't want to use the multi-scale gradient matching term !!!
        # regularization_loss = self.regularization_loss(inputs.squeeze(1), targets.squeeze(1), masks.squeeze(1))
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)

        # =============================
        n, c, h, w = targets.size()
        if n != 1:
            inputs = inputs.view(n, -1).double()
            targets = targets.view(n, -1).double()
            masks = masks.view(n, -1).double()
            edges_img = edges_img.view(n, -1).double()
            thetas_img = thetas_img.view(n, -1).double()

        else:
            inputs = inputs.contiguous().view(1, -1).double()
            targets = targets.contiguous().view(1, -1).double()
            masks = masks.contiguous().view(1, -1).double()
            edges_img = edges_img.contiguous().view(1, -1).double()
            thetas_img = thetas_img.contiguous().view(1, -1).double()

        # initialization
        loss = torch.DoubleTensor([0.0]).to(self.device)

        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num = self.edgeGuidedSampling(
                inputs[i, :], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)
            # Random Sampling
            random_sample_num = sample_num
            random_inputs_A, random_inputs_B, random_targets_A, random_targets_B, random_masks_A, random_masks_B = randomSampling(
                inputs[i, :], targets[i, :], masks[i, :], self.min_threshold, random_sample_num)

            # Combine EGS + RS
            inputs_A = torch.cat((inputs_A, random_inputs_A), 0)
            inputs_B = torch.cat((inputs_B, random_inputs_B), 0)
            targets_A = torch.cat((targets_A, random_targets_A), 0)
            targets_B = torch.cat((targets_B, random_targets_B), 0)
            masks_A = torch.cat((masks_A, random_masks_A), 0)
            masks_B = torch.cat((masks_B, random_masks_B), 0)

            # GT ordinal relationship
            target_ratio = torch.div(targets_A + 1e-6, targets_B + 1e-6)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0 / (1.0 + self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0 / (1.0 + self.sigma))] = -1

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A * masks_B

            equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() * consistency_mask
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (
                ~mask_eq).double() * consistency_mask

            # Please comment the regularization term if you don't want to use the multi-scale gradient matching d_loss !!!
            loss = loss + self.alpha * equal_loss.mean() + 1.0 * unequal_loss.mean()  # + 0.2 * regularization_loss.double()

        return loss[0].float() / n

    def edgeGuidedSampling(self, inputs, targets, edges_img, thetas_img, masks, h, w):

        # find edges
        edges_max = edges_img.max()
        edges_mask = edges_img.ge(edges_max * 0.1)
        edges_loc = edges_mask.nonzero()

        inputs_edge = torch.masked_select(inputs, edges_mask)
        targets_edge = torch.masked_select(targets, edges_mask)
        thetas_edge = torch.masked_select(thetas_img, edges_mask)
        minlen = inputs_edge.size()[0]

        # find anchor points (i.e, edge points)
        sample_num = minlen
        index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long, device=self.device)
        anchors = torch.gather(inputs_edge, 0, index_anchors)
        theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
        row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
        ## compute the coordinates of 4-points,  distances are from [2, 30]
        distance_matrix = torch.randint(2, 31, (4, sample_num)).to(self.device)
        pos_or_neg = torch.ones(4, sample_num).to(self.device)
        pos_or_neg[:2, :] = -pos_or_neg[:2, :]
        distance_matrix = distance_matrix.float() * pos_or_neg
        col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(
            distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
        row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(
            distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()

        # constrain 0=<c<=w, 0<=r<=h
        # Note: index should minus 1
        col[col < 0] = 0
        col[col > w - 1] = w - 1
        row[row < 0] = 0
        row[row > h - 1] = h - 1

        # a-b, b-c, c-d
        a = sub2ind(row[0, :], col[0, :], w)
        b = sub2ind(row[1, :], col[1, :], w)
        c = sub2ind(row[2, :], col[2, :], w)
        d = sub2ind(row[3, :], col[3, :], w)
        A = torch.cat((a, b, c), 0)
        B = torch.cat((b, c, d), 0)

        inputs_A = torch.gather(inputs, 0, A.long())
        inputs_B = torch.gather(inputs, 0, B.long())
        targets_A = torch.gather(targets, 0, A.long())
        targets_B = torch.gather(targets, 0, B.long())
        masks_A = torch.gather(masks, 0, A.long())
        masks_B = torch.gather(masks, 0, B.long())

        return inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num
