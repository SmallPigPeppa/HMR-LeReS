from leres_model_pl import LeReS
from hmr_leres_config import args as train_args
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    depth_model = LeReS()
    callbacks = []
    wandb_logger = WandbLogger(name='leres-v5.0', project='HMR-LeReS0-v5.0', entity='pigpeppa', offline=False)
    wandb_logger.watch(depth_model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(train_args)
    checkpoint_callback = ModelCheckpoint(dirpath="leres-ckpt-v5.0", save_last=True, save_top_k=2, monitor="total_loss")

    trainer = Trainer(
        gradient_clip_val=10.0,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), checkpoint_callback],
    )

    trainer.fit(depth_model)

