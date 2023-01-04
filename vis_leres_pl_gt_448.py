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
    leres_model = LeReS()
    depth_model = leres_model.depth_model.eval()
    image_dir_out = 'leres-vis-out'
    image_name = 'test-gt-448-559'
    image_input = os.path.join(image_dir_out, '00559.jpg')
    image = cv2.imread(image_input)

    leres_loader = leres_model.train_dataloader()['leres_loader']
    depth = next(iter(leres_loader))['depth'].numpy().squeeze()
    # depth = depth - depth.min()
    dmax = np.percentile(depth, 90)
    dmin = np.percentile(depth, 10)
    # depth = np.maximum(depth, dmin)
    depth = np.minimum(depth, dmax)
    depth_ori = cv2.resize(depth, (image.shape[1], image.shape[0]),interpolation=cv2.INTER_NEAREST)
    # depth_ori=depth

    rgb = next(iter(leres_loader))['rgb'].numpy().squeeze()
    rgb = np.transpose(rgb, (2,1, 0))
    rgb_ori = cv2.resize(rgb, (image.shape[1], image.shape[0]))

    focal_length = 1.15803374e+03
    reconstruct_depth(depth_ori, image[:,:,::-1], image_dir_out, image_name + '-pcd-2', focal=focal_length)
