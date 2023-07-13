# conda activate hmr-leres
export PYTHONPATH="./HMR/src:./LeReS/Train"
/root/miniconda3/envs/hmr-leres/bin/python main_all_plane_origin_align_10801920.py \
--smpl-mean-theta-path HMR/HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt \
--gta_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--mesh_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--ckpt_dir hmr-leres-ckpt-debug/10801920 \
--name v8.0-10801920 \
--project HMR-LeReS-v8.0 \
--entity pigpeppa \
--num_gpus 8 \
--max-epochs 100 \
--num-workers 8 \
--batch-size 16 \
--eval-batch-size 16 \
--adv-batch-size 16 \
--weight_decay 2e-5 \
--base_lr 0.1 \
--e_lr 0.01 \
--d-lr 0.01





/root/miniconda3/envs/hmr-leres/bin/python main_all_plane_origin_align_fix_pretrain.py \
--smpl-mean-theta-path HMR/HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt \
--gta_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--mesh_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--ckpt_dir hmr-leres-ckpt-debug/fix-pretrain-448 \
--name fix-pretrain-448 \
--project HMR-LeReS-v8.0 \
--entity pigpeppa \
--num_gpus 8 \
--max-epochs 100 \
--num-workers 8 \
--batch-size 16 \
--eval-batch-size 16 \
--adv-batch-size 16 \
--weight_decay 2e-5 \
--base_lr 0.1 \
--e_lr 0.01 \
--d-lr 0.01


/root/miniconda3/envs/hmr-leres/bin/python main_all_plane_origin_align_no_pretrain.py \
--smpl-mean-theta-path HMR/HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR/HMR-data/neutral_smpl_with_cocoplus_reg.txt \
--gta_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--mesh_dataset_dir /mnt/mmtech01/dataset/vision_text_pretrain/gta-im \
--ckpt_dir hmr-leres-ckpt-debug/no-pretrain-448 \
--name no-pretrain-448 \
--project HMR-LeReS-v8.0 \
--entity pigpeppa \
--num_gpus 8 \
--max-epochs 100 \
--num-workers 8 \
--batch-size 16 \
--eval-batch-size 16 \
--adv-batch-size 16 \
--weight_decay 2e-5 \
--base_lr 0.1 \
--e_lr 0.01 \
--d-lr 0.01



#--e_lr 0.00001
#--d-lr 0.0001
#--base_lr 0.001
