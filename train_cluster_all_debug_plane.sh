# conda activate hmr-leres
export PYTHONPATH="./HMR/src:./LeReS/Train"
/root/miniconda3/envs/hmr-leres/bin/python main_all_plane.py \
--smpl-mean-theta-path HMR/HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt \
--gta_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--mesh_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--ckpt_dir hmr-leres-ckpt-debug \
--name hmr-leres-cluster \
--project HMR-LeReS-v7.0-debug \
--entity pigpeppa \
--num_gpus 8 \
--max-epochs 100 \
--num-workers 8 \
--batch-size 1 \
--eval-batch-size 16 \
--adv-batch-size 48 \
--weight_decay 0.0005
#--e_lr 0.00001
#--d-lr 0.0001
#--base_lr 0.001
