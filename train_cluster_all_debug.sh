# conda activate hmr-leres
export PYTHONPATH="./HMR/src:./LeReS/Train"
/share/wenzhuoliu/conda-envs/hmr-leres-new/bin/python main_all.py \
--smpl-mean-theta-path HMR/HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt \
--gta_dataset_dir /share/wenzhuoliu/torch_ds/gta-im \
--mesh_dataset_dir /share/wenzhuoliu/torch_ds/gta-im \
--ckpt_dir hmr-leres-ckpt-debug \
--name hmr-leres-cluster \
--project HMR-LeReS-v7.0-debug \
--entity pigpeppa \
--num_gpus 2 \
--max-epochs 100 \
--num-workers 8 \
--batch-size 16 \
--eval-batch-size 16 \
--adv-batch-size 48 \
--weight_decay 0.0005
#--e_lr 0.00001
#--d-lr 0.0001
#--base_lr 0.001
