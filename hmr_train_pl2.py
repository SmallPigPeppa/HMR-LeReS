from hmr_model_pl import HMR
from hmr_leres_config import args as train_args
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    seed_everything(5)
    hmr_model=HMR()
    callbacks=[]
    wandb_logger = WandbLogger(
        name='hmr-v6.0', project='HMR-LeReS-v6.0', entity='pigpeppa', offline=False
    )
    wandb_logger.watch(hmr_model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(train_args)

    checkpoint_callback = ModelCheckpoint(dirpath="hmr-ckpt-v6.0",save_last=True, save_top_k=2, monitor="loss_generator")

    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10),checkpoint_callback],
    )

    trainer.fit(hmr_model)
