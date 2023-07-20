from hmr_leres_model_all_fast_debug_plane_origin_align_fix_pretrain_kptsfilter import HMRLeReS
from hmr_leres_config import args
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    seed_everything(5)
    model = HMRLeReS()
    model.leres_model.load_ckpt('/mnt/mmtech01/usr/liuwenzhuo/code/HMR-LeReS-v8/a_models/leres_pretrain/res50.pth')
    callbacks = []
    wandb_logger = WandbLogger(
        name=args.name, project=args.project, entity=args.entity, offline=args.offline
    )
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=2, monitor="save_ckpt_loss")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=args.num_gpus,
        strategy='ddp',
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), checkpoint_callback]
    )
    # log_every_n_steps = 1,

    trainer.fit(model)
