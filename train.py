import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd

import pytorch_lightning as pl

from swin import *
from dataset import *


pl.seed_everything(3407, workers=True)

def training_classification():
    df = pd.read_json('/path/to/rsna_miccai_radiogenomics_cross_validation.json')
    for fold in range(5):
        train_subjects, train_transform, val_subjects, val_transform = rsna_miccai_radiogenomics_cross_validation(dataframe=df, fold=fold)
        train_loader, val_loader = dataloader_cross_validation(train_subjects, train_transform, val_subjects, val_transform, batch_size=1, num_workers=4)

        model = swin_baseline()
        wandb_logger = pl.loggers.WandbLogger(project="neuroradiology")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_acc", 
            dirpath="/path/to/checkpoints/rsna_miccai_radiogenomics_fold_{}".format(fold), 
            filename="swin-classification-{fold:02d}-{epoch:02d}-{val_loss:.6f}-{val_acc:.6f}-{val_f1:.6f}-{val_pr:.6f}-{val_re:.6f}", 
            mode="max",
        )

        trainer = pl.Trainer(
            gpus=-1, 
            max_epochs=100, 
            log_every_n_steps=5,
            gradient_clip_val=0.5, 
            callbacks=[
                pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
                checkpoint_callback,
                ], 
            gradient_clip_algorithm="value",
            logger=wandb_logger,
            )
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":    
    training_classification()
