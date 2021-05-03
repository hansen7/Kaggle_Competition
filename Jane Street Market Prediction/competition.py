
import time as time_log
import pickle

import pandas as pd
import numpy as np
import torch

from trend_net import TrendClassifier
from utility_net import UtilityMaximizer


# Loop that iterates through each trade opportunity that comes up in real time and sends an action to the Jane Street API

def competition_loop(utility_max=0):


    print('loading models...')

    trend_models = []

    torch.manual_seed(0)
    np.random.seed(0)

    for cv_no in np.arange(4):

        trend_model = TrendClassifier.load_from_checkpoint('./weights/trend_CV4'+str(cv_no)+'.ckpt', input_width=136)
        trend_model.cpu()
        trend_model.eval()

        trend_models.append(trend_model)

    utility_models = []

    for cv_no in np.arange(4):

        utility_model = UtilityMaximizer.load_from_checkpoint('./weights/utility_CV4'+str(cv_no)+'.ckpt', input_width=134)
        utility_model.cpu()
        utility_model.eval()

        utility_models.append(utility_model)


    print('loading data...')

    ## ## ## ## ## ## ## remove in kaggle notebook  ## ## ## ## ## ## ## ##
    data = pd.read_csv('D:/data/jane-street/train.csv', nrows=10000)
    data = data.iloc[:,np.concatenate(([1],np.arange(7,7+130),[0]))]  # as provided by Jane Street API
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

    medians = pickle.load(open('./preprocessing/medians.pkl','rb'))
    scaler = pickle.load(open('./preprocessing/scaler.pkl','rb'))
    clf_day_trend = pickle.load(open('./preprocessing/clf_day_trend.pkl','rb'))
    clf_day_volat = pickle.load(open('./preprocessing/clf_day_volat.pkl','rb'))



    print('loop starts...')

    cache_len = 100
    cache_features = np.zeros((2,cache_len,130), dtype=np.float32)
    cache_features[0] = medians[0]  #broadcasting
    cache_features[1] = medians[1]
    cache_idx = np.array([0, 0])

    times = []
    very_start = time_log.time()

    for i in np.arange(len(data)):
    #for (test_df, sample_prediction_df) in iter_test:

        start_loop = time_log.time()

        # as provided by Jane Street API: weight, features, day
        entry = data.iloc[i].to_numpy(dtype=np.float32).squeeze()
        #entry = test_df.to_numpy(dtype=np.float32).squeeze()

        weight = entry[0]
        features = entry[1:-1]
        side = features[0]
        side_id = 0 if side == -1 else 1

        # imputation
        no_imputation_required = ((np.sum(np.isnan(features)) == 0)*1).astype(dtype=np.float32)

        imputed = np.nanmedian(cache_features[side_id], axis=0)

        features = np.nan_to_num(features) + np.isnan(features) * np.nan_to_num(imputed)
        assert(np.isinf(features).sum() == 0)
        assert(np.isnan(features).sum() == 0)

        # cache update
        cache_features[side_id,cache_idx[side_id]] = features
        cache_idx[side_id] = (cache_idx[side_id]+1)%cache_len

        if weight == 0:
            times.append((time_log.time() - start_loop)*1000)
            this_goes_to_jane_street_API = 0
            #sample_prediction_df.action = this_goes_to_jane_street_API
            #env.predict(sample_prediction_df)
            continue

        # Z-scores
        chunk_val = cache_features[side_id]
        chunk_mea = np.nanmean(chunk_val, axis=0)
        chunk_std = np.nanstd(chunk_val, ddof=1, axis=0)

        # std < 1e-6 is ignored for stability (with numbers close to 0, Z-scores go bananas)
        tozero = np.isnan(chunk_std) | (chunk_std < 1e-3)

        mask = np.ones(chunk_mea.shape, dtype=np.float32)
        mask[tozero] = 0
        chunk_mea[tozero] = 0
        chunk_std[tozero] = 1

        Z_scores = (features*mask - chunk_mea) / chunk_std

        Z_scores[0] = features[0]
        Z_scores[64] = features[64]

        assert(np.isinf(Z_scores).sum() == 0)
        assert(np.isnan(Z_scores).sum() == 0)

        # interactions
        interactions = np.column_stack((    
                                            features[3] * features[45],
                                            features[10] * features[122],
                                            features[14] * features[58],
                                            features[22] * features[42],
                                            features[35] * features[20],
                                            features[45] * features[47],
                                        ))

        features = np.concatenate(( features, interactions.squeeze() ))

        # normalization
        features[1:] = scaler.transform(features[1:].reshape(1,-1)).squeeze()

        # network models

        chunk_trend = clf_day_trend.predict(chunk_mea.reshape(1,-1)).item()
        chunk_volat = clf_day_volat.predict(chunk_mea.reshape(1,-1)).item()

        final_pred = 0.

        for cv_no in np.arange(4):

            if not utility_max:
                trend_model_input = torch.tensor(features, device='cpu').reshape(1,-1)
                final_pred += torch.sigmoid(trend_models[cv_no](trend_model_input)).item()

            else:
                utility_model_input = torch.tensor(np.concatenate((  Z_scores,
                                                                    np.array([no_imputation_required]),
                                                                    np.array([weight]), 
                                                                    np.array([chunk_trend], dtype=np.float32),
                                                                    np.array([chunk_volat], dtype=np.float32) ))
                                                                    , device='cpu').reshape(1,-1)

            final_pred += torch.sigmoid(utility_models[cv_no](utility_model_input)).item()

        this_goes_to_jane_street_API = ((final_pred/4) > 0.5)*1
        #sample_prediction_df.action = this_goes_to_jane_street_API
        #env.predict(sample_prediction_df)

        times.append((time_log.time() - start_loop)*1000)

    print('avg iteration time:', round(np.mean(times),2),'ms')
    print('total loop time:', round(time_log.time() - very_start),'seconds')


if __name__ == '__main__':

    np.set_printoptions(threshold=2000, linewidth=130, precision=3, edgeitems=20, suppress=1)

    competition_loop()
