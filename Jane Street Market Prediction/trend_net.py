
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score

from utils import blend, smooth_bce, utility_score


class TrendClassifier(LightningModule):

    def __init__(self, input_width):

        super(TrendClassifier, self).__init__()

        hidden_size = input_width*2

        self.batch_norm1 = nn.BatchNorm1d(input_width)
        self.dense1 = nn.Linear(input_width, hidden_size)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.35)

        x = self.batch_norm2(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.4)

        x = self.batch_norm3(x)
        x = self.dense3(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.45)

        return self.linear(x)

    def training_step(self, train_batch, batch_idx):

        x, _, context, weights_nn = train_batch

        blended_x, blended_c, blended_w = blend(x, context, weights_nn)

        logits = self.forward(blended_x)
        resp_mean = torch.mean(blended_c[:,2:7], axis=1)

        target = (resp_mean > 0).float()

        loss = smooth_bce(logits, target, smoothing=0.0, weight=blended_w.squeeze())

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        x, trend_class, context, _ = val_batch

        logits = self.forward(x)
        loss = smooth_bce(logits, trend_class, smoothing=0.0)

        with torch.no_grad():

            pred = torch.sigmoid(logits)

            val_auc = roc_auc_score(y_true=trend_class.detach().cpu().squeeze(),
                                    y_score=pred.detach().cpu().squeeze())

            _, _, val_u = utility_score(context, (pred>0.5).float())
            print('U:',np.round(val_u,2))

            self.log('val_loss', loss)
            self.log('val_u', val_u)
            self.log('val_auc', val_auc)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=1/5, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_auc'}


    @staticmethod
    def predict(X, split):

        path = './weights/trend_'+split+'.ckpt'

        if os.path.isfile(path):

            tre_model = TrendClassifier.load_from_checkpoint(path, input_width=X.shape[1])
            tre_model.cpu()
            tre_model.eval()

            tre_test = torch.tensor(X, dtype=torch.float32, requires_grad=False, device='cpu')
            return torch.sigmoid(tre_model(tre_test)).detach().numpy().squeeze()

        return None


    @staticmethod
    def predict_blend(X):

        trend_pred = None

        if os.path.isdir('./weights/trend/'):

            tre_files = walk('./weights/trend/')

            root, _, tre_model_paths = next(tre_files)
            tre_test = torch.tensor(X, dtype=torch.float32, requires_grad=False, device='cpu')
            tre_num_models = 0

            for model_path in tre_model_paths:

                tre_num_models += 1

                if tre_num_models == 1:
                    trend_pred = np.zeros((tre_test.shape[0])).astype(np.float32) 

                tre_model = TrendClassifier.load_from_checkpoint(root+model_path, input_width=X.shape[1])
                tre_model.cpu()
                tre_model.eval()

                cvpred = torch.sigmoid(tre_model(tre_test)).detach().numpy().squeeze()
                trend_pred += cvpred

            trend_pred /= tre_num_models

        return trend_pred

