from HMR.src.dataloader.gta_dataloader import gta_dataloader as hmr_dataset
from LeReS.Train.data.gta_dataset import GTADataset as leres_dataset
from torch.utils.data import Dataset



class MergeGTADataset(Dataset):
    def __init__(self, opt,dataset_name, data_set_path, use_crop, scale_range, use_flip, min_pts_required, pix_format='NHWC',
                 normalize=False, flip_prob=0.3):
        super(MergeGTADataset, self).__init__()
        self.hmr_dataset = hmr_dataset(data_set_path, use_crop, scale_range, use_flip, min_pts_required, pix_format,
                                       normalize, flip_prob)
        self.leres_dataset = leres_dataset(opt, dataset_name)

    def __getitem__(self, index):
        hmr_input = self.hmr_dataset[index]
        leres_input = self.leres_dataset[index]
        hmr_input.update(leres_input)
        return hmr_input

    def __len__(self):
        assert len(self.leres_dataset) == len(self.hmr_dataset)
        return len(self.leres_dataset)


if __name__ == '__main__':
    from merge_config import args

    data_set_path='C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
    pix_format = 'NCHW'
    normalize = True
    flip_prob = 0.5
    use_flip = False
    mdataset = MergeGTADataset(opt=args,
                               dataset_name='2020-06-11-10-06-48',
                               data_set_path=data_set_path,
                               use_crop=True,
                               scale_range=[1.1, 2.0],
                               use_flip=True,
                               min_pts_required=5,
                               pix_format=pix_format,
                               normalize=normalize,
                               flip_prob=flip_prob, )
    # print('mdataset.leres_dataset[0]',mdataset.leres_dataset[0])
    # print('mdataset.hmr_dataset[0]',mdataset.hmr_dataset[0])
    item=mdataset[0]
    a=item['kp_2d']*224+112
    print('mdataset[0]',mdataset[0])
    print('len(mdataset)',len(mdataset))
