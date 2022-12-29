import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from collections import OrderedDict

from LeReS.Minist_Test.lib_test.test_utils import refine_focal, refine_shift
from LeReS.Minist_Test.lib_test.multi_depth_model_woauxi import RelDepthModel
from LeReS.Minist_Test.lib_test.spvcnn_classsification import SPVCNN_CLASSIFICATION
from LeReS.Minist_Test.lib_test.test_utils import reconstruct_depth
from leres_model_pl import LeReS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args()
    return args


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def make_shift_focallength_models():
    shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    shift_model.eval()
    focal_model.eval()
    return shift_model, focal_model


def reconstruct3D_from_depth(rgb, pred_depth, shift_model, focal_model):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax

    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60 / 2.0) * np.pi / 180))

    # recover focal
    focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()

    # recover shift
    shift_1 = refine_shift(pred_depth_norm, shift_model, predicted_focal_1, cam_u0, cam_v0)
    shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
    depth_scale_1 = pred_depth_norm - shift_1.item()

    # recover focal
    focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

    return shift_1, predicted_focal_2, depth_scale_1


def load_ckpt(ckpt_path, depth_model, shift_model, focal_model):
    """
    Load checkpoint.
    """
    if os.path.isfile(ckpt_path):
        print("loading checkpoint %s" % ckpt_path)
        checkpoint = torch.load(ckpt_path)
        if shift_model is not None:
            shift_model.load_state_dict(strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                        strict=True)
        if focal_model is not None:
            focal_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                        strict=True)
        depth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                                    strict=True)
        del checkpoint
        torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


if __name__ == '__main__':
    pl_ckpt_path = 'leres-ckpt-backup/last.ckpt'
    leres_model = LeReS.load_from_checkpoint(pl_ckpt_path)
    depth_model = leres_model.depth_model.eval()
    loader = leres_model.train_dataloader()
    data = next(iter(loader['leres_loader']))
    rgb = data['rgb']
    depth, _ = depth_model(rgb)

    sf_ckpt_path = 'res50.pth'
    # # create depth model
    # depth_model = RelDepthModel(backbone='resnet50')
    # depth_model.eval()

    # create shift and focal length model
    shift_model, focal_model = make_shift_focallength_models()
    # load checkpoint
    sf_ckpt = torch.load(sf_ckpt_path)
    shift_model.load_state_dict(strip_prefix_if_present(sf_ckpt['shift_model'], 'module.'),
                                strict=True)
    focal_model.load_state_dict(strip_prefix_if_present(sf_ckpt['focal_model'], 'module.'),
                                strict=True)
    # depth_model.cuda()
    shift_model.cuda()
    focal_model.cuda()

    image_dir_out = 'leres_vis_out'
    image_name = 'demo-559'
    rgb=rgb[0]
    depth=depth[0].permute(2, 0, 1)
    print('depth.shape',depth.shape)
    pred_depth_out = depth - depth.min() + 0.01
    pred_depth = pred_depth_out.cpu().detach().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # recover focal length, shift, and scale-invariant depth
    shift, focal_length, depth_scaleinv = reconstruct3D_from_depth(rgb, pred_depth_ori,
                                                                   shift_model, focal_model)

    # if GT depth is available, uncomment the following part to recover the metric depth
    # pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

    # save depth
    plt.imsave(image_name + '-depth.png', pred_depth_ori, cmap='rainbow')
    cv2.imwrite(image_name + '-depth_raw.png', (pred_depth_ori / pred_depth_ori.max() * 60000).astype(np.uint16))
    # save disp


    print(depth_scaleinv.shape)
    print(rgb.shape)
    reconstruct_depth(depth_scaleinv, rgb.numpy()[:, :, ::-1], image_dir_out, image_name + '-pcd', focal=focal_length)
