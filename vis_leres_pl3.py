import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms

from LeReS.Minist_Test.lib_test.test_utils import reconstruct_depth
from leres_model_pl import LeReS




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


if __name__ == '__main__':
    # pl_ckpt_path = 'leres-ckpt-backup/last.ckpt'
    pl_ckpt_path = 'leres-ckpt/last.ckpt'
    leres_model = LeReS.load_from_checkpoint(pl_ckpt_path)
    depth_model = leres_model.depth_model.eval()
    image = leres_model.train_dataloader()

    image_dir_out = 'leres-vis-out'
    image_name = 'demo-559'
    # '/share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48/00559.jpg'
    image_input = os.path.join(image_dir_out, '00559.jpg')
    rgb = cv2.imread(image_input)
    rgb_c = rgb[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (448, 448))

    img_torch = scale_torch(A_resize)[None, :, :, :]
    # pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    depth, _ = depth_model(img_torch)

    # depth = depth - depth.min() + 0.01
    depth = depth - depth.min()
    # pred_depth_out = depth - depth.min()
    depth = depth.cpu().detach().numpy().squeeze()
    dmax = np.percentile(depth, 95)
    dmin = np.percentile(depth, 5)
    depth = np.maximum(depth, dmin)
    depth = np.minimum(depth, dmax)
    depth_ori = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

    focal_length = 1.15803374e+03
    # focal_length = 1500

    reconstruct_depth(depth_ori, rgb[:, :, ::-1], image_dir_out, image_name + '-pcd', focal=focal_length)
