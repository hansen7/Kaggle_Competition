
import pickle
import os.path
import time
import gc

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset, Sampler
import pytorch_lightning as pl

from trend_net import TrendClassifier
from utility_net import UtilityMaximizer

from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score

# for debugging
np.set_printoptions(threshold=2000, linewidth=110, precision=3, edgeitems=20, suppress=1)


class JaneStreetDataModule(pl.LightningDataModule):

    def __init__(self, model=None, batch_size=4096, split='CV40', num_rows=None, percent_rows=None):

        super(JaneStreetDataModule, self).__init__()

        self.split = split
        self.batch_size = batch_size

        features, context, Z_scores, no_imput_req, weights_nn, market_trend, market_volat, set_label = np.repeat(None, 8)

        file_path = 'D:/data/jane-street/cache/'+split+'.pkl'

        if os.path.isfile(file_path):
            
            print('loading data from cache file...')

            with open(file_path,'rb') as f:

                array = pickle.load(f).astype(np.float32)

                features = array[:,:136]
                context = array[:,136 : 136+8]
                Z_scores = array[:,136+8 : 136+8+130]
                no_imput_req = array[:,-5]
                weights_nn = array[:,-4]
                market_trend = array[:,-3]
                market_volat = array[:,-2]
                is_train = array[:,-1]

                set_label = np.repeat('train', len(features))
                set_label[is_train==0] = 'val'

        else:

            features, context, Z_scores, no_imput_req, weights_nn, \
                market_trend, market_volat, set_label = self.load_data(num_rows, percent_rows)

            print('saving file cache...')

            is_train = (set_label=='train').astype(np.float32)

            with open('D:/data/jane-street/cache/'+split+'.pkl','wb') as f:
                pickle.dump(np.hstack(( features, context, Z_scores,
                                        no_imput_req.reshape(-1,1),
                                        weights_nn.reshape(-1,1),
                                        market_trend.reshape(-1,1),
                                        market_volat.reshape(-1,1),
                                        is_train.reshape(-1,1) )), f, protocol=4)

            gc.collect()

        print('creating datasets...')

        print('train size:', np.sum(set_label=="train"))

        self.train_ds = JaneStreetDataSet(  model,
                                            features[set_label=='train'],
                                            context[set_label=='train'],
                                            Z_scores[set_label=='train'],
                                            no_imput_req[set_label=='train'],
                                            weights_nn[set_label=='train'],
                                            market_trend[set_label=='train'],
                                            market_volat[set_label=='train']
                                          )


        print('val size:', np.sum(set_label=="val"))

        self.val_ds = JaneStreetDataSet(model,
                                            features[set_label=='val'],
                                            context[set_label=='val'],
                                            Z_scores[set_label=='val'],
                                            no_imput_req[set_label=='val'],
                                            weights_nn[set_label=='val'],
                                            market_trend[set_label=='val'],
                                            market_volat[set_label=='val']
                                        )


    def load_data(self, num_rows=None, percent_rows=None):

        start_prepro = time.time()

        print('reading data...')

        if num_rows is None:
            file_data = pd.read_csv('D:/data/jane-street/train.csv')
        else:
            file_data = pd.read_csv('D:/data/jane-street/train.csv', nrows=num_rows)

        if not percent_rows is None:
            np.random.seed(0)
            r = np.random.choice([True, False], p =[percent_rows, 1-percent_rows], size=len(file_data))
            file_data = file_data[r]

        features = file_data.iloc[:,7:7+130].to_numpy(dtype=np.float32)

        # 'date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp', 'ts_id'
        context = file_data.iloc[:,np.concatenate((np.arange(7),[-1]))].to_numpy(dtype=np.float32)


        print('lasso regime models...')

        # Regression with binary classes - the neural network will make sense of the values
        # All the 500 days included in the training

        file_data['market_trend'] = -100 * file_data.feature_0 * file_data.resp

        days = file_data.groupby('date').mean()
        days['market_vol'] = file_data.groupby('date')[['market_trend']].std()

        days['market_trend'] = (days['market_trend'] > 0)*1
        days['market_vol'] = (days['market_vol'] > days['market_vol'].median())*1

        day_features = days.iloc[:,6:6+130]

        clf_day_trend = LassoCV(cv=5, random_state=0, max_iter=50000)
        clf_day_trend.fit(day_features, days['market_trend'])
        pickle.dump(clf_day_trend, open('./preprocessing/clf_day_trend.pkl','wb'))

        print('clf_day_trend.coef_', clf_day_trend.coef_)
        print('clf_day_trend.intercept_', clf_day_trend.intercept_)

        clf_day_volat = LassoCV(cv=5, random_state=0, max_iter=50000)
        clf_day_volat.fit(day_features, days['market_vol'])
        pickle.dump(clf_day_volat, open('./preprocessing/clf_day_volat.pkl','wb'))

        print('clf_day_vol.coef_', clf_day_volat.coef_)
        print('clf_day_vol.coef_', clf_day_volat.intercept_)

        file_data = None
        gc.collect()


        print('split...')

        sides = features[:,0]
        days = context[:,0]

        np.random.seed(0)

        if self.split.startswith('CV'):
            jump = int(self.split[-2])
            shift = int(self.split[-1])
            set_label = np.where((days + shift) % jump  == 0, 'val', 'train')
        elif self.split.startswith('mixed'):
            perc = int(self.split[-2:])
            set_label = np.random.choice(['train', 'val'], p =[1-perc/100, perc/100], size=len(features))
        else:
            raise Exception('Unknown split method')

        medians = np.vstack((   np.nanmedian(features[(set_label=='train') & (sides==-1)], axis=0),
                                np.nanmedian(features[(set_label=='train') & (sides==+1)], axis=0) ))

        with open('./preprocessing/medians.pkl','wb') as f:
            pickle.dump(medians, f)


        print('imputation and Z-scores...')

        no_imput_req = ((np.sum(np.isnan(features), axis=1) == 0)*1).astype(dtype=np.float32)

        cache_len = 100
        cache_pointer = np.array([0, 0])
        cache_features = np.zeros((2,cache_len,features.shape[1]), dtype=np.float32)
        cache_features[0] = medians[0]  #broadcasting
        cache_features[1] = medians[1]

        Z_scores = np.zeros(features.shape, dtype=np.float32)

        market_trend = np.zeros(len(features), dtype=np.float32)
        market_volat = np.zeros(len(features), dtype=np.float32)

        for i in np.arange(len(features)):

            # imputation
            side_id = 0 if features[i,0] == -1 else 1
            cache_mea = np.nanmean(cache_features[side_id], axis=0)
            cache_std = np.nanstd(cache_features[side_id], ddof=1, axis=0)

            features[i] = np.nan_to_num(features[i]) + np.isnan(features[i]) * cache_mea

            # Z-score
            nope = np.isnan(cache_std) | (cache_std < 1e-3)

            mask = np.ones(cache_mea.shape)
            mask[nope] = 0
            cache_mea[nope] = 0
            cache_std[nope] = 1

            Z_scores[i] = (features[i]*mask - cache_mea) / cache_std

            # market trend and volatility
            market_trend[i] = clf_day_trend.predict(cache_mea.reshape(1,-1))
            market_volat[i] = clf_day_volat.predict(cache_mea.reshape(1,-1))

            # cache update
            cache_features[side_id,cache_pointer[side_id]] = features[i]
            cache_pointer[side_id] = (cache_pointer[side_id]+1) % cache_len

            if i % 1000 == 0:
                print(round(i*100/len(features),1),'%', end='\r')

        Z_scores[:,0] = features[:,0]
        Z_scores[:,64] = features[:,64]

        assert(np.isnan(features).sum() == 0)
        assert(np.isinf(features).sum() == 0)

        assert (np.isnan(Z_scores).sum() == 0)
        assert (np.isinf(Z_scores).sum() == 0)


        print('interactions, normalization and weights...')

        interactions = np.column_stack((    
                                            features[:,3] * features[:,45],
                                            features[:,10] * features[:,122],
                                            features[:,14] * features[:,58],
                                            features[:,22] * features[:,42],
                                            features[:,35] * features[:,20],
                                            features[:,45] * features[:,47],
                                        ))

        features = np.hstack(( features, interactions ))

        scaler = StandardScaler().fit(features[set_label=="train",1:])
        pickle.dump(scaler, open('./preprocessing/scaler.pkl','wb'))
        features[:,1:] = scaler.transform(features[:,1:])

        weights_js = context[:,1]
        
        weights_nn = weights_js * np.abs(np.mean(context[:,2:7], axis=1))
        weights_nn[weights_nn < 1e-5] = 0.0
        weights_nn[weights_nn > 0.4] = 0.4

        assert(np.isnan(Z_scores).sum() == 0)
        assert(np.isinf(Z_scores).sum() == 0)
        assert (np.isnan(features).sum() == 0)
        assert (np.isinf(features).sum() == 0)
        assert (np.isnan(context).sum() == 0)
        assert (np.isinf(context).sum() == 0)
        assert (np.isnan(market_trend).sum() == 0)
        assert (np.isinf(market_trend).sum() == 0)
        assert (np.isnan(market_volat).sum() == 0)
        assert (np.isinf(market_volat).sum() == 0)


        print('loading and preprocessing done in',round(time.time() - start_prepro),'seconds')

        return features, context, Z_scores, no_imput_req, weights_nn, market_trend, market_volat, set_label


    def setup(self, stage):

        pass

    def train_dataloader(self):

        if self.train_ds.model == UtilityMaximizer:

            return DataLoader(self.train_ds, batch_sampler=JaneStreetSampler(self.train_ds, num_batches=20*4))

        else:
            return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):

        if self.train_ds.model == UtilityMaximizer:

            return DataLoader(self.val_ds, batch_size=len(self.val_ds.X), shuffle=True)

        else:

            return DataLoader(self.val_ds, batch_size=len(self.val_ds.X), shuffle=False)


class JaneStreetDataSet(Dataset):
    
    def __init__(self, model, features, context, Z_scores, no_imput_req, weights_nn, market_trend, market_volat):

        super(Dataset, self).__init__()

        self.model = model

        # context columns: 'date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp', 'ts_id'
        self.context = context
        self.trend_class = (np.mean(context[:,2:7],axis=1) > 0).astype(dtype=np.float32).reshape(-1,1)
        self.weights_nn = weights_nn.reshape(-1,1)

        if self.model == TrendClassifier:

            self.X = features

        elif self.model == UtilityMaximizer:

            self.X = np.hstack((    Z_scores,
                                    no_imput_req.reshape(-1,1),
                                    context[:,1].reshape(-1,1),
                                    market_trend.reshape(-1,1),
                                    market_volat.reshape(-1,1)
                                ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ndx):

        return self.X[ndx], self.trend_class[ndx], self.context[ndx], self.weights_nn[ndx]



class JaneStreetSampler(Sampler):

    ### Splits the trade opportunities in 2 sets of days

    def __init__(self, dataset, num_batches = 20):

        self.X = dataset.X
        self.c = dataset.context
        self.num_batches = num_batches

    def __iter__(self):

        rand_idx = np.random.permutation(len(self.X))
        days_idx = np.random.permutation(500)

        block_size = len(rand_idx) // (self.num_batches//2)
        leftover = len(rand_idx) % (self.num_batches//2)

        all_idx = np.arange(len(rand_idx))

        for i in np.arange(self.num_batches//2):

            for j in np.arange(2):

                batch_rand = rand_idx[block_size*i : block_size*(i+1) if i < 9 else block_size*(i+1) + leftover]
                batch_days = days_idx[250*j : 250*(j+1)]

                idx = np.where(np.isin(self.c[:,0], batch_days) & np.isin(all_idx, batch_rand))[0]                    

                np.random.shuffle(idx)
                yield  idx

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':

    data = JaneStreetDataModule()





