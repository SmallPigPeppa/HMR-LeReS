# conda activate HMR-LeReS3
export PYTHONPATH="/share/wenzhuoliu/code/HMR-LeReS-new/HMR/src:/share/wenzhuoliu/code/HMR-LeReS-new/LeReS/Train"
python h_main_train_cluster.py \
--smpl-mean-theta-path HMR/HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt \
--gta_dataset_dir /share/wenzhuoliu/torch_ds/HMR-LeReS/2020-06-11-10-06-48 \
--mesh_dataset_dir /share/wenzhuoliu/torch_ds/HMR-LeReS/mosh \
--ckpt_dir hmr-leres-ckpt \
--name hmr-leres-cluster \
--project HMR-LeReS-v6.0 \
--entity pigpeppa \
--num_gpus 2 \
--max-epochs 100 \
--num-workers 0 \
--batch-size 4 \
--eval-batch-size 4 \
--adv-batch-size 24 \
--weight_decay 0.0005
