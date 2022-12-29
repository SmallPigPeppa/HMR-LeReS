export PYTHONPATH="/share/wenzhuoliu/code/HMR-LeReS/HMR/src:/share/wenzhuoliu/code/HMR-LeReS/LeReS/Train"
#export PYTHONPATH="/share/wenzhuoliu/code/HMR-LeReS/LeReS/Train"
/share/wenzhuoliu/conda-envs/HMR-LeReS/bin/python leres_train_pl.py \
--smpl-mean-theta-path HMR/HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt \
--save-folder HMR/HMR-data/out-model \
--num-worker 0 \
--dataset gta \
--dataroot /share/wenzhuoliu/torch_ds/HMR-LeReS \
--backbone resnet50 \
--dataset_list 2020-06-11-10-06-48 \
--batchsize 2 \
--base_lr 0.001 \
--use_tfboard \
--thread 0 \
--loss_mode _ranking-edge_pairwise-normal-regress-edge_msgil-normal_meanstd-tanh_pairwise-normal-regress-plane_ranking-edge-auxi_meanstd-tanh-auxi_ \
--epoch 2 \
--lr_scheduler_multiepochs 1 \
--val_step 10000 \
--snapshot_iters 100000 \
--log_interval 1