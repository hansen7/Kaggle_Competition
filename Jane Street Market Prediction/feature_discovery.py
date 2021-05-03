import torch
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler

from data import JaneStreetDataModule


if __name__ == '__main__':

    data = JaneStreetDataModule(model=TabNetClassifier, split='mixed10')

    X_t = data.train_ds.X
    X_v = data.val_ds.X

    y_t = data.train_ds.trend_class.astype(dtype=np.int64).squeeze()
    y_v = data.val_ds.trend_class.astype(dtype=np.int64).squeeze()

    for i in np.arange(X_t.shape[1]):

        interactions_t = X_t * np.repeat(X_t[:,i:(i+1)],X_t.shape[1],axis=1)
        interactions_v = X_v * np.repeat(X_v[:,i:(i+1)],X_v.shape[1],axis=1)

        scaler = StandardScaler().fit(interactions_t)
        interactions_t = scaler.transform(interactions_t)
        interactions_v = scaler.transform(interactions_v)

        torch.manual_seed(0)
        np.random.seed(0)

        clf = TabNetClassifier( scheduler_params={"step_size":5, "gamma":0.8},
                                scheduler_fn=torch.optim.lr_scheduler.StepLR )

        clf.fit(
            interactions_t, y_t,
            eval_set=[(interactions_t, y_t),(interactions_v, y_v)],
            eval_name=['train','val'],
            batch_size = 4096,
            virtual_batch_size = 4096//8,
            patience = 3,
            max_epochs = 100
        )

        # print('importances:',clf.feature_importances_*100)

        selected = np.where(clf.feature_importances_*100 > 8)[0]
        print('multiplying by:',i)
        print('selected:',selected)
        print('percents:',clf.feature_importances_[selected]*100)
        print()

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

    clf = TabNetClassifier( n_d=8,
                            n_a=8,
                            n_steps=3,
                            optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=2e-2),
                            scheduler_params={"step_size":6, "gamma":0.9},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR )

    clf.fit(
        X_t, y_t,
        eval_set=[(X_t, y_t),(X_v, y_v)],
        eval_name=['train','val'],
        batch_size = 4096,
        virtual_batch_size = 4096//8,
        patience = 7,
        max_epochs = 16
    )

    print()
    print('importances:',clf.feature_importances_*100)

    print()
    selected = np.where(clf.feature_importances_*100 > 1)[0]
    print('selected:',selected)

    print()
    print('percents:',clf.feature_importances_[selected]*100)
