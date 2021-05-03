from datetime import datetime

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data import JaneStreetDataModule
from trend_net import TrendClassifier
from utility_net import UtilityMaximizer

def fit_model(model, percent_rows=None, num_rows=None, split='mixed10', batch_size=4096):

    data = JaneStreetDataModule(model=model, split=split, batch_size=batch_size, num_rows=num_rows, percent_rows=percent_rows)

    model_width = data.train_ds.X.shape[1]
    print('model_width',model_width)

    if model == TrendClassifier:
        monitor = 'val_auc'
        model = TrendClassifier(model_width)
        filename = 'trend-'+split+'-{epoch}-{val_auc:.4f}-{val_u:.4f}'
        dirpath='./weights/'
    elif model == UtilityMaximizer:
        monitor = 'val_u'
        model = UtilityMaximizer(model_width)
        filename = 'utility-'+split+'-{epoch}-{val_auc:.2f}-{val_u:.4f}'
        dirpath='./weights/'
    else:
        raise NotImplementedError()

    print('model:',model)
    print('monitoring:',monitor)
    print('time start:',datetime.now().strftime("%H:%M:%S"))

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        patience=7,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor=monitor,
        mode='max'
    )

    trainer = pl.Trainer(   logger=pl_loggers.TensorBoardLogger('./logs/'),
                            gpus=1,
                            max_epochs=1000,
                            checkpoint_callback=checkpoint_callback,
                            callbacks=[early_stop_callback] )

    torch.manual_seed(0)
    np.random.seed(0)

    trainer.fit(model, data)

    print('time end:',datetime.now().strftime("%H:%M:%S"))


if __name__ == '__main__':

    for i in np.arange(4):
        fit_model(TrendClassifier, split='CV4'+str(i), batch_size=4096)
    
    for i in np.arange(4):
        fit_model(UtilityMaximizer, split='CV4'+str(i))

