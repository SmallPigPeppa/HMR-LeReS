from hmr_model_pl import HMR
from hmr_leres_config import args as train_args
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    hmr_model=HMR()
    callbacks=[]
    wandb_logger = WandbLogger(
        name='hmr-test', project='HMR-LeReS', entity='pigpeppa', offline=False
    )
    wandb_logger.watch(hmr_model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(train_args)

    checkpoint_callback = ModelCheckpoint(dirpath="hmr-ckpt",save_last=True, save_top_k=2, monitor="e_loss")

    trainer = Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10),checkpoint_callback],
    )

    trainer.fit(hmr_model)

