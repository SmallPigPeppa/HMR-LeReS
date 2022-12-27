from leres_model_pl import LeReS
from leres_model_pl import ModelOptimizer
from merge_config import args as train_args
from merge_gta_dataset import MergeGTADataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

if __name__ == '__main__':
    depth_model = LeReS()
    # depth_model = torch.nn.DataParallel(depth_model.cuda())
    data_set_path = 'C:/Users/90532/Desktop/Datasets/HMR/2020-06-11-10-06-48'
    pix_format = 'NCHW'
    normalize = True
    flip_prob = 0
    use_flip = False
    opt = train_args
    merged_dataset = MergeGTADataset(opt=train_args,
                                     dataset_name='2020-06-11-10-06-48',
                                     data_set_path=data_set_path,
                                     use_crop=True,
                                     scale_range=[1.1, 2.0],
                                     use_flip=False,
                                     min_pts_required=5,
                                     pix_format=pix_format,
                                     normalize=normalize,
                                     flip_prob=flip_prob, )
    dataloader = DataLoader(
        dataset=merged_dataset,
        batch_size=opt.batchsize,
        num_workers=opt.thread,
        shuffle=True,
        drop_last=True,
        pin_memory=True, )
    start_step = 0
    total_iters = 1000
    optimizer = ModelOptimizer(depth_model)
    for step in tqdm(range(start_step, total_iters)):
        data = next(iter(dataloader))
        out = depth_model(data)
        losses_dict = out['losses']
        optimizer.optim(losses_dict)
