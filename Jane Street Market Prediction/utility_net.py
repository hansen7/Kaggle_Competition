import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score

from utils import blend, utility_score


class UtilityMaximizer(LightningModule):

    def __init__(self, input_width):

        super(UtilityMaximizer, self).__init__()

        self.m_batch_norm1 = nn.BatchNorm1d(input_width)
        self.m_dense1 = nn.Linear(input_width, input_width*2)

        self.m_batch_norm2 = nn.BatchNorm1d(input_width*2)
        self.m_dense2 = nn.Linear(input_width*2, input_width*2)

        self.m_batch_norm3 = nn.BatchNorm1d(input_width*2)
        self.m_dense3 = nn.Linear(input_width*2, input_width*2)

        self.linear = nn.Linear(input_width*2, 1)

    def forward(self, x):

        x = self.m_batch_norm1(x)
        x = self.m_dense1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.35)

        x = self.m_batch_norm2(x)
        x = self.m_dense2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.4)

        x = self.m_batch_norm3(x)
        x = self.m_dense3(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.45)

        return self.linear(x)


    def training_step(self, train_batch, batch_idx):

        x, _, context, w = train_batch

        blended_x, blended_c, _ = blend(x, context, w)

        logits = self.forward(blended_x)
        pred = torch.sigmoid(logits)
        loss = utility_score(blended_c, pred, mode='loss')

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        x, trend_class, context, _ = val_batch
        logits = self.forward(x)
        pred = torch.sigmoid(logits)
        loss = utility_score(context, pred, mode='loss')
        targets = (trend_class > 0).float()

        with torch.no_grad():

            val_auc = roc_auc_score(y_true=targets.detach().cpu(), y_score=pred.detach().cpu())

            val_t, val_p, val_u = utility_score(context, (pred>0.5).float())

            self.log('val_loss', loss)
            self.log('val_auc', val_auc)

            self.log('val_t', val_t)
            self.log('val_p', val_p)
            self.log('val_u', val_u)

            return val_auc, val_t, val_p, val_u

    def validation_epoch_end(self, val_step_outputs):

        ma = np.array(val_step_outputs)

        print()
        print()
        print('TOTAL auc:', np.round(np.mean(ma[:,0],axis=0),3))
        print('TOTAL val_t:', np.round(np.mean(ma[:,1],axis=0),2))
        print('TOTAL val_p:', np.round(np.mean(ma[:,2],axis=0),2))
        print('TOTAL val_u:', np.round(np.mean(ma[:,3],axis=0),2))

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=1/5, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_u'}


