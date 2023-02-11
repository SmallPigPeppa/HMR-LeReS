import argparse

parser = argparse.ArgumentParser(description='HMR-LeReS model')

# HMR
parser.add_argument('--smpl-mean-theta-path', type=str, default='E:/HMR/model/neutral_smpl_mean_params.h5',
                    help='the path for mean smpl theta value')
parser.add_argument('--smpl-model', type=str, default='E:/HMR/model/neutral_smpl_with_cocoplus_reg.txt',
                    help='smpl model path')
parser.add_argument('--smpl-model-dir', type=str, default='HMR/HMR-data/smpl',
                    help='smpl model path')
parser.add_argument('--total-theta-count', type=int, default=85, help='the count of theta param')
parser.add_argument('--joint-count', type=int, default=24, help='the count of joints')
parser.add_argument('--beta-count', type=int, default=10, help='the count of beta')
parser.add_argument('--use-adv-train', type=bool, default=True, help='use adv traing or not')
parser.add_argument('--scale-min', type=float, default=1.1, help='min scale')
parser.add_argument('--scale-max', type=float, default=1.5, help='max scale')
parser.add_argument('--e-lr', type=float, default=0.00001, help='encoder learning rate.')
parser.add_argument('--d-lr', type=float, default=0.0001, help='Adversarial prior learning rate.')
parser.add_argument('--e-wd', type=float, default=0.0001, help='encoder weight decay rate.')
parser.add_argument('--d-wd', type=float, default=0.0001, help='Adversarial prior weight decay')
parser.add_argument('--e-d_loss-weight', type=float, default=60, help='weight on encoder 2d kp losses.')
parser.add_argument('--d-d_loss-weight', type=float, default=1, help='weight on discriminator losses')
parser.add_argument('--d-disc-ratio', type=float, default=1.0, help='multiple weight of discriminator d_loss')
parser.add_argument('--e-3d-d_loss-weight', type=float, default=60, help='weight on encoder thetas losses.')
parser.add_argument('--e-shape-ratio', type=float, default=5, help='multiple weight of shape d_loss')
parser.add_argument('--e-3d-kp-ratio', type=float, default=10.0, help='multiple weight of 3d key point.')
parser.add_argument('--e-pose-ratio', type=float, default=20, help='multiple weight of pose')
parser.add_argument('--enable-inter-supervision', type=bool, default=False, help='enable inter supervision or not.')
parser.add_argument('--encoder-network', type=str, default='resnet50', help='the encoder network name')
parser.add_argument('--batch-size', type=int, default=4, help='batch size')
parser.add_argument('--batch-3d-size', type=int, default=8, help='3d data batch size')
parser.add_argument('--eval-batch-size', type=int, default=4, help='3d data batch size')
parser.add_argument('--adv-batch-size', type=int, default=24, help='default adv batch size')

# LeReS
parser.add_argument('--backbone', type=str, default='resnet50', help='Select backbone type, resnet50 or resnext101')
parser.add_argument('--batchsize', type=int, default=4, help='Batch size')
parser.add_argument('--base_lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--dataroot', default='./datasets', help='Path to images')
parser.add_argument('--loss_mode', default='_vnl_ssil_ranking_', help='Select d_loss to supervise, joint or ranking')
parser.add_argument('--lr_scheduler_multiepochs', default=[10, 25, 40], nargs='+', type=int,
                    help='Learning rate scheduler step')
parser.add_argument('--scale_decoder_lr', type=float, default=1, help='Scale learning rate for the decoder')
parser.add_argument('--sample_ratio_steps', default=10000, type=int, help='Step for increasing sample ratio')
parser.add_argument('--sample_start_ratio', default=0.1, type=float, help='Start sample ratio')
parser.add_argument('--phase', type=str, default='train', help='Training flag')
parser.add_argument('--phase_anno', type=str, default='train', help='Annotations file name')

# Shared
parser.add_argument('--epoch', default=50, type=int, help='Total training epochs')
parser.add_argument('--num-worker', type=int, default=1, help='pytorch number worker.')




train_2d_set = ['gta']
train_3d_set = ['gta']
train_adv_set = ['mosh']

eval_set = ['up3d']

allowed_encoder_net = ['hourglass', 'resnet50', 'densenet169']

encoder_feature_count = {
    'hourglass': 4096,
    'resnet50': 2048,
    'densenet169': 1664
}

crop_size = {
    'hourglass': 256,
    'resnet50': 224,
    'densenet169': 224
}

data_set_path = {
    'coco': 'E:/HMR/data/COCO/',
    'lsp': 'E:/HMR/data/lsp',
    'lsp_ext': 'E:/HMR/data/lsp_ext',
    'ai-ch': 'E:/HMR/data/ai_challenger_keypoint_train_20170902',
    'mpi-inf-3dhp': 'E:/HMR/data/mpi_inf_3dhp',
    'hum3.6m': 'E:/HMR/data/human3.6m',
    'mosh': 'C:/Users/90532/Desktop/Datasets/HMR/mosh',
    'gta': 'C:/Users/90532/Desktop/Datasets/HMR-LeReS/2020-06-11-10-06-48'
}

pre_trained_model = {
    'generator': '/media/disk1/zhangxiong/HMR/hmr_resnet50/fine_tuned/3500_generator.pkl',
    'discriminator': '/media/disk1/zhangxiong/HMR/hmr_resnet50/fine_tuned/3500_discriminator.pkl'
}

args = parser.parse_args()
encoder_network = args.encoder_network
args.feature_count = encoder_feature_count[encoder_network]
args.crop_size = crop_size[encoder_network]
