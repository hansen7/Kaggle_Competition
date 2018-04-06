# [1. Algorithmic Trading Challenge](https://www.kaggle.com/c/AlgorithmicTradingChallenge)

###1.1 Overview

**Goal**: 

- The aim of this competition is to determine the relationship between recent past order book events and future stock price mean reversion following shocks to liquidity.

**Submission Format(Things to be predicted)** : 

- For each observation a participant should provide 100 numbers describing the next 50 bid and ask prices(from t51 to t100) following a liquidity shock. There should be 50,001 rows in total (50,000 entries plus a header row) and 101 columns for each row.

**Evaluation Method** : 

- Root Mean Square Error(RMSE) ：$\sqrt{\frac{\Sigma_{t=1}^n(\hat{y_t}-y_t)^2}{n}}$, where $\hat{y_t}$ and $y_t$ are the predicted/actual value respectively.

### 1.2 Data

The competition host provide 4 files for preparation，train.csv & test.csv, plus example_entry_example.csv&example_entry_naive.csv, the last two seems to be used as submission template, though I can't tell their difference. But it seems that these two files have similiar dimension, and differences seems to have patterns. Here's the comparison using pandas.DataFrame：

```python
In [26]: ex_entry_linear == ex_entry_naive
Out[26]: 
       row_id  bid51  ask51  bid52  ask52  bid53  ask53  bid54  ask54  bid55  \
0        True  False   True  False   True  False   True  False   True  False   
1        True   True  False   True  False   True  False   True  False   True   
2        True  False   True  False   True  False   True  False   True  False   
3        True   True  False   True  False   True  False   True  False   True   
4        True   True  False   True  False   True  False   True  False   True   
5        True  False   True  False   True  False   True  False   True  False   
6        True  False   True  False   True  False   True  False   True  False   
7        True  False   True  False   True  False   True  False   True  False   
8        True   True  False   True  False   True  False   True  False   True   
9        True   True  False   True  False   True  False   True  False   True   
10       True   True  False   True  False   True  False   True  False   True   
11       True   True  False   True  False   True  False   True  False   True   
12       True   True  False   True  False   True  False   True  False   True   
13       True  False   True  False   True  False   True  False   True  False   
14       True   True  False   True  False   True  False   True  False   True
```



Anyway, the meaning of provided labels in training/test set are shown as below：

![](https://raw.githubusercontent.com/hansen7/Kaggle_Competition/master/Algorithmic_Trading_Challenge/f1.png)



![](https://raw.githubusercontent.com/hansen7/Kaggle_Competition/master/Algorithmic_Trading_Challenge/f2.png)



The Size of each files are listed:

```python
##train.csv
In [10]: training_set.index
Out[10]: RangeIndex(start=0, stop=754018, step=1)

##test.csv
In [22]: test_set.index
Out[22]: RangeIndex(start=0, stop=50000, step=1)    
    
##example_entry_linear.csv
In [13]: ex_entry_linear.index
Out[13]: RangeIndex(start=0, stop=50000, step=1)

##example_entry_naive.csv
In [16]: ex_entry_naive.index
Out[16]: RangeIndex(start=0, stop=50000, step=1)
```



### 1.3 Selected Solutions

**Key Methods: **Time Partitioning Prediction, Random Forest, Ensemble Methods and Feature Extractions.

### 1.3.1 [Top1 Solution](http://www.ms.k.u-tokyo.ac.jp/2013/Kaggle.pdf) 

#### Results: 

- Ranked 1st in the private leaderboard, 4th in the public leaderboard.

#### Work Flow:

- **Step1: Time Interval Partitioning Algorithm**, Due to identity between t=50 and t=51, the partition start from t=52 to t=100. Use greedy algorithm to make partition(when the error in algorithm step7 begins to rise, then stop and separate ), and use different sub-models($M_{bit}(t)$  & $M_{ask}(t)$  for bit/ask prediction respectively). Furthermore, assume the length of later segmentation is larger than the earlier ones(length($C_{i+1}$) > length($C_i$)), this is derived from the main hypothesis of the model:

  ```reStructuredText
  'an always increasing prediction error may require averaging longer price time series to obtain a constant price prediction with an acceptable error' 
  													-Cited from author
  ```

  Different $M_{bit, i}(t)$  & $M_{ask, i}(t)$ are used to describe each time periods($C_i$):![](https://raw.githubusercontent.com/hansen7/Kaggle_Competition/master/Algorithmic_Trading_Challenge/figure4.png)

  ![](https://raw.githubusercontent.com/hansen7/Kaggle_Competition/master/Algorithmic_Trading_Challenge/f3.png)

- **Step2: Feature Extraction**, The author have extracted over 150 features(divided into 4 classes: *Price, Liquidity book, Spread, Rate*) from the original data, using a [specified R module](http:// cran.r-project.org/web/packages/TTR/TTR.pdf]) (I guess)

  ![](https://raw.githubusercontent.com/hansen7/Kaggle_Competition/master/Algorithmic_Trading_Challenge/figure5.png)



- **Step3: Modeling Approach Selection,** First, using MIC(Maximal Information Coefficient ) to analyze the mutual infromation between features and the dependent variable, revealing a very low mutual information and non-functional relationships; then tried both Random Forest and Gradient Boosting Machines to select the data.(4-cross validation, and the **refined training set** consisted of 1.same size 2. same combination of security_id as the test set, then choose RF for lower RMSE cost. )



####

- **Step4: Feature Selection**, inspired by a similiar method applied to the [*Kaggle*'s Heritage Health Prize dataset.](https://www.kaggle.com/c/hhp) The algorithm are divided into 2 parts(1. step1 - 8: to get the quasi-optimized $S_f$ quickly, 2. Rest of the steps: to make local adjustments on these $S_f$ feature set ).

  ![](https://raw.githubusercontent.com/hansen7/Kaggle_Competition/master/Algorithmic_Trading_Challenge/f6.png)

  ​

- **Step5: Validation**: 

  - **Feature Set** Validation: Ironically, the author used the same $S_f$ set for both $F_b$ and $F_a$ (bit and ask feature set for bit and ask price respectively) at the final stage due to the lack of time for calcu dlation. **BUT, for the different time period, the $F_b$ is different.**
  - **Optimal Time Segmentation**: The final partition of the time period is {t=52, t=53, t=54–55, t=56–58, t=59–64, t=65–73, t=74–100}, which I thought quite make sense: the nearest period has the highest weight, so is sepearted.


- **Model Performance via methods**:  

  Work Flow:

####Comment 

- Pros: 
- Cons:

### 1.4 Comment

这题数据是由London Stocks Exchange提供的，较为原始，也有较明晰的参考资料，虽然是用R语言写的，亮点是将需要预测的时间进行分段，不过这篇solution很早，估计可能time interval partition是现在的主流做法了。。后面的竞赛evaluation用的是Weighted Mean Absolute Error。

#  

# [2. Benchmark Bond Trade Price Challenge](https://www.kaggle.com/c/benchmark-bond-trade-price-challenge)

### 2.1 Overview

**Goal** The aim of this competition: **To Predict the next price that a US corporate bond might trade at.** Contestants are given information on the bond including current coupon, time to maturity and a reference price computed by [Benchmark Solutions](http://www.benchmarksolutions.com/).  Details of the previous 10 trades are also provided.  

**Evaluation Method**: Mean Absolute Error *with Weighted Factors(Square Root of the time since the last observation + Normalization)* 这也就意味着最近时间的结果 所占的权重最大



### 2.2 Data

基本上就提供了train.csv和test.csv两个文件，和一个用来benchmark的R文件

Column details:

- id: The row id. 
- bond_id: The unique id of a bond to aid in timeseries reconstruction. (This column is only present in the train data)
- trade_price: The price at which the trade occured.  (This is the column to predict in the test data)
- **weight**: The weight of the row for evaluation purposes. This is calculated as the square root of the time since the last trade and then scaled so the mean is 1. 
- current_coupon: The coupon of the bond at the time of the trade.
- **time_to_maturity**: The number of years until the bond matures at the time of the trade.
- **is_callable**: A binary value indicating whether or not the bond is callable by the issuer.
- **reporting_delay**: The number of seconds after the trade occured that it was reported.
- trade_size: The notional amount of the trade.
- **trade_type**: 2=customer sell, 3=customer buy, 4=trade between dealers. We would expect customers to get worse prices on average than dealers. 
- curve_based_price: A fair price estimate based on implied hazard and funding curves of the issuer of the bond.
- received_time_diff_last{1-10}: The time difference between the trade and that of the previous {1-10}.
- trade_price_last{1-10}: The trade price of the last {1-10} trades.
- trade_size_last{1-10}: The notional amount of the last {1-10} trades.
- trade_type_last{1-10}: The trade type of the last {1-10} trades.
- curve_based_price_last{1-10}: The curve based price of the last {1-10} trades.

```python
In [33]: train_set.index
Out[33]: RangeIndex(start=0, stop=762678, step=1)

In [34]: train_set.columns
Out[34]: 
Index(['id', 'bond_id', 'trade_price', 'weight', 'current_coupon',
       'time_to_maturity', 'is_callable', 'reporting_delay', 'trade_size','trade_type', 'curve_based_price', 'received_time_diff_last1',
       'trade_price_last1', 'trade_size_last1', 'trade_type_last1',
       'curve_based_price_last1', 'received_time_diff_last2',
       'trade_price_last2', 'trade_size_last2', 'trade_type_last2',
       'curve_based_price_last2', 'received_time_diff_last3',
       'trade_price_last3', 'trade_size_last3', 'trade_type_last3',
       'curve_based_price_last3', 'received_time_diff_last4',
       'trade_price_last4', 'trade_size_last4', 'trade_type_last4',
       'curve_based_price_last4', 'received_time_diff_last5',
       'trade_price_last5', 'trade_size_last5', 'trade_type_last5',
       'curve_based_price_last5', 'received_time_diff_last6',
       'trade_price_last6', 'trade_size_last6', 'trade_type_last6',
       'curve_based_price_last6', 'received_time_diff_last7',
       'trade_price_last7', 'trade_size_last7', 'trade_type_last7',
       'curve_based_price_last7', 'received_time_diff_last8',
       'trade_price_last8', 'trade_size_last8', 'trade_type_last8',
       'curve_based_price_last8', 'received_time_diff_last9',
       'trade_price_last9', 'trade_size_last9', 'trade_type_last9',
       'curve_based_price_last9', 'received_time_diff_last10',
       'trade_price_last10', 'trade_size_last10', 'trade_type_last10',
       'curve_based_price_last10'],
      dtype='object')

In [37]: train_set.columns.size
Out[37]: 61
```



### 2.3 Winner's Solution [Top 2(Sergey Yurgenson)'s Comment on Kernel](https://www.kaggle.com/c/benchmark-bond-trade-price-challenge/discussion/1833)

Key Methods: **Random Forest(随机森林)，GBM(???)**

*Other Useful Tutorials can be found here: [1](https://www.slideshare.net/RanZhang14/benchmark-the-actual-bond-prices?from_action=save), [2](http://novieq.blogspot.com/2013/10/benchmark-bond-trade-price-challenge.html), more details can be found on the attached pdf file*





1. Results
2. Methods
3. Comments(Pros and Cons)











































### 2.4 Comment

这题原始数据是由股票公司提供的，**高频交易**，因此有很多extracted feature，更偏向于实际情况，feature selection情况也会更加复杂，解答资料也不明晰。



# [3. The Winton Stock Market Challenge](https://www.kaggle.com/c/the-winton-stock-market-challenge)

### 3.1 Overview

**Goal** To Find the hidden signal in the terabytes of noisy, non-stationary data via novel statistical modelling and data mining techniques. In this competition the challenge is **to predict the return of a stock, given the history of the past few days.**

**Evaluation Method**: Weighted Mean Absolute Error *Weighted Factors is associated with the return* 所用的评价体系跟第二题一样，权重计算方法不同。

### 3.2 Data

基本上就提供了train.csv和test.csv两个文件，和一个用来参考作为submission format的csv



![mage-20180401224149](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804012241497.png)

We provide 5-day windows of time, days D-2, D-1, D, D+1, and D+2. You are given returns in days D-2, D-1, and part of day D, and you are asked to predict the returns in the rest of day D, and in days D+1 and D+2.

During day D, there is intraday return data, which are the returns at different points in the day. We provide 180 minutes of data, from t=1 to t=180. In the training set you are given the full 180 minutes, in the test set just the first 120 minutes are provided.

For each 5-day window, we also provide 25 features, Feature_1 to Feature_25. These may or may not be useful in your prediction.（更加抽象的feature）

Each row in the dataset is an arbitrary stock at an arbitrary 5 day time window.

（一支stock占一行，要求开发一个能够在预测所有stock走势的model？）



More can be visited [here](https://www.kaggle.com/c/the-winton-stock-market-challenge/data).



### 3.3 Top Solution [Top 2 Discussion](https://www.kaggle.com/c/the-winton-stock-market-challenge/discussion/18591) [Top 15 Discussion](https://www.kaggle.com/c/the-winton-stock-market-challenge/discussion/18584)

Key Methods: **bayesian linear regression**, 没有特别明确的解答，这个问题也是**高频交易**



### 3.4 Comment

这题feature更加抽象，**高频交易**，解答资料基本没有。





# [4. Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting)

### 4.1 Overview

**Goal** To forecast the future values of multiple time series. In this competition the challenge is **to predict the return of a stock, given the history of the past few days.**

**Evaluation Method**: [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)(Symmetric mean absolute percentage error)。

**Highlight** two stages of competition, second part is based on real-time data.

### 4.2 Data

**train_\*.csv** - contains traffic data. This a csv file where each row corresponds to a particular article and each column correspond to a particular date. Some entries are missing data. The page names contain the *Wikipedia project (e.g. en.wikipedia.org), type of access (e.g. desktop) and type of agent (e.g. spider).*

In other words, each article name has the following format: 'name_project_access_agent' (e.g. 'AKB48_zh.wikipedia.org_all-access_spider').

**key_\*.csv** - gives the mapping between the page names and the shortened Id column used for prediction

**sample_submission_\*.csv** - a submission file showing the correct format

```python
In [5]: train_set.index
Out[5]: RangeIndex(start=0, stop=145063, step=1)

In [7]: train_set.columns
Out[7]: 
Index(['Page', '2015-07-01', '2015-07-02', '2015-07-03', '2015-07-04',
       '2015-07-05', '2015-07-06', '2015-07-07', '2015-07-08', '2015-07-09',
       ...
       '2016-12-22', '2016-12-23', '2016-12-24', '2016-12-25', '2016-12-26',
       '2016-12-27', '2016-12-28', '2016-12-29', '2016-12-30', '2016-12-31'],
      dtype='object', length=551)
```

**so 550 daily traffic data of 145063 Wikepedia sites for the training set**

### 4.3 Top Solution [Top 1 Solution with Code](https://github.com/Arturus/kaggle-web-traffic),  [Top 2 Discussion](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/39395), [Top 6 With Code](https://github.com/sjvasquez/web-traffic-forecasting)

Key Methods: RNN seq2seq, 就这些github上已经非常全了，以下主要概括Top 1的solution

#### 4.3.1 Information Sources for Prediction

- Local features. If we see a trend, we expect that it will continue **(AutoRegressive model)**, if we see a traffic spike, it will gradually decay **(Moving Average model)**, if wee see more traffic on holidays, we expect to have more traffic on holidays in the future **(seasonal model)**.
- Global features. If we look to autocorrelation plot, we'll notice strong year-to-year autocorrelation and some quarter-to-quarter autocorrelation.

The good model should use both global and local features, combining them in a intelligent way.

#### 4.3.2 Model Selection - RNN seq2seq

1. RNN can be thought as a natural extension of well-studied [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)(Autoregressive integrated moving average) models, but much more flexible and expressive.
2. RNN is non-parametric, that's greatly **simplifies learning**. Imagine working with different ARIMA parameters for 145K timeseries.
3. Any exogenous feature (numerical or categorical, time-dependent or series-dependent) can be easily injected into the model
4. seq2seq seems natural for this task: we predict next values, conditioning on joint probability of previous values, including our past predictions. Use of past predictions stabilizes the model, it learns to be conservative, because error accumulates on each step, and extreme prediction at one step can ruin prediction quality for all subsequent steps.
5. Deep Learning is all the hype nowadays     —> RNN。。。

#### 4.3.3 Feature Engineering

I tried to be minimalistic, because **RNN is powerful enough to discover and learn features on its own.** Model feature list:

- *pageviews* (spelled as 'hits' in the model code, because of my web-analytics background). Raw values transformed by log1p() to get more-or-less normal intra-series values distribution, instead of skewed one.
- *agent*, *country*, *site* - these features are extracted from page urls and one-hot encoded
- *day of week* - to capture weekly seasonality
- *year-to-year autocorrelation*, *quarter-to-quarter autocorrelation* - to capture yearly and quarterly seasonality strength.
- *page popularity* - High traffic and low traffic pages have different traffic change patterns, this feature (median of pageviews) helps to capture traffic scale. This scale information is lost in a *pageviews* feature, because each pageviews series independently normalized to zero mean and unit variance.
- ***lagged pageviews* - I'll describe this feature later**



####4.3.4 Feature Preprocessing

- All features (including one-hot encoded) are **normalized** to zero mean and unit variance. Each *pageviews* series normalized independently.
- Time-independent features (autocorrelations, country, etc) are "stretched" to timeseries length i.e. repeated for each day by `tf.tile()` command.
- Model trains on random fixed-length samples from original timeseries. For example, if original timeseries length is 600 days, and we use 200-day samples for training, we'll have a choice of 400 days to start the sample.

This sampling works as effective data augmentation mechanism: training code randomly chooses starting point for each timeseries on each step, generating endless stream of almost non-repeating data.



#### 4.3.5 Model Core

![mage-20180402165649](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804021656497.png)

#### 4.3.6 Working with long timeseries

- LSTM/GRU能够很好的处理短序列的问题(100-300items)，但比赛中涉及到超过700天的数据，因此作者采用了一些方法来‘strengthen’ GRU memory.

  - First trial is called 'attention' method:

    ![mage-20180402202639](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804022026398.png)

    I can just take encoder outputs from `current_day - 365` and `current_day - 90` timepoints, pass them through FC layer to reduce dimensionality and append result to input features for decoder. This solution, despite of being simple, **considerably lowered prediction error.**

  - Then, to reduce the noise: `attn_365 = 0.25 * day_364 + 0.5 * day_365 + 0.25 * day_366`

  - Then I realized that `0.25,0.5,0.25` is a 1D convolutional kernel (length=3) and I can automatically learn bigger kernel to detect important points in a past.（非常6）

  - Ended up with a monstrous attention mechanism not classical method such as **(Bahdanau or Luong attention)**, because of calculation cost.

  - 但其实最好的public score反倒是最简单的模型：

    ![mage-20180402204822](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804022048228.png)

  - 另外发现60-90day的encoder表现也还okay

#### 4.3.7 Redefine Loss Function

- [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) (target loss for competition) can't be used directly, because of unstable behavior near zero values (loss is a step function if truth value is zero, and not defined, if predicted value is also zero).
- Candidates include *differentiable SMAPE* and *MAE*

#### 4.3.8 Training and validation

- [COCOB](https://arxiv.org/abs/1705.07795) optimizer

- Prefer walk-forward split rather than side-by-side split

  ![mage-20180402210447](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804022104479.png)

#### 4.3.9 Reducing model variance

- Performance varies from training steps and seed.
- Solutions:
  - But I know approximate region where model is (possibly) trained well enough, but (possibly) not started to overfit. I decided to set this optimal region bounds to 10500..11500 training steps and save 10 checkpoints from each 100th step in this region.
  - Similarly, I decided to train 3 models on different seeds and save checkpoints from each model. So I have 30 checkpoints total.
  - One widely known method for reducing variance and improving model performance is SGD averaging (ASGD). Method is very simple and well supported in [Tensorflow](https://www.tensorflow.org/versions/r0.12/api_docs/python/train/moving_averages) - we have to maintain moving averages of network weights during training and use these averaged weights, instead of original ones, during inference.
  - 然后 AVERAGE！！！

#### 4.3.10 Hyperparameter Tuning

略略略。。。

### 4.4 Comment

solution set非常丰富，1st place有着详细的code和readme，用的是RNN，有一些地方feature extraction不清晰，可以先做。



# [5. Two Sigma Financial Modeling Challenge](https://www.kaggle.com/c/two-sigma-financial-modeling)

### 5.1 Overview

**Goal** To forecast the future values of multiple time series. In this competition the challenge is **to predict the return of a stock, given the history of the past few days.**

**Evaluation Method**: R Value(signed square root of R^2 value), Negative R values are clipped at -1, i.e. the score you see will be max(−1,R)max(−1,R)

![mage-20180401234614](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804012346141.png)

![mage-20180401234602](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804012346027.png)



### 5.2 Data

**train_\*.csv** 就这一个文件，我kaggle的api(应该是kaggle不支持python3.6?)出现了一些问题，暂时没下载下来数据，但是可以看到很多data interpretation在discussion session里面



### 5.3 Top Solution [Top 12 Solution with Code](https://www.kaggle.com/c/two-sigma-financial-modeling/discussion/29518), [Some Exploration](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-5)

Key Methods: **Ensemble Learning** Ridge 和 拉搜

Our solution (Giba and Xpeuler) is a basic blend of 7 models:

1. Ridge A - Selected Features trained on SignedExp( y+1 ). Some cleaning on features and filter on target instances.
2. Ridge B - Selected features and some cleaning on features and filter on target instances.
3. Ridge C - Selected features and some cleaning on features and filter on target instances. 
4. Extra Trees - Selected features. 222 trees
5. XGB - Selected Features and tuned hyperparameters on "all" trainset.
6. Ridge online rolling fit - Trained every 100 steps on submission time. features: Some lags of [technical20-technical30]. Target used: lag1 of [technical20-technical30] 
7. Variance by Step(day) - Simple variance calculated over all 'Id' per day 

The final predictions are a weighted average of that 7 models.

Crossvalidation for model performance and feature selection was made using some approaches:

- 2 folds: timestamp > 906 and timestamp <= 906
- 5 kfolds
- rolling fit for ts> 906

### 5.4 Comment

solution set不是很丰富，我看他们的解答都各种说“black magic”。。。





# [6. Predict short term movements in stock prices using news and sentiment data provided by RavenPack](https://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge)

### 6.1 Overview

**Goal**  To develop a model that predicts stock price movements using sentiment data provided by RavenPack.

**Evaluation Method**: mean absolute error

![mage-20180402004611](/var/folders/c_/q1g90c_s3d712_h2cljszmx40000gp/T/abnerworks.Typora/image-201804020046110.png)

### 6.2 Data

**data.zip** - contains features for 510 days worth of trading, including 200 training days and 310 testing days
**trainLabels.csv** - contains the targets for the 200 training days
**sampleSubmission.csv** - shows the submission format

Each variable named O1, O2, O3, etc. (the outputs) represents a percent change in the value of a security.  Each variable named I1, I2, I3, etc. (the inputs) represents a feature.*（所以这里还是把sentiment已经抽象成了feature）* The underlying securities and features represented by these anonymized names are the same across all files (e.g. O1 will always be the same stock)...

### 6.3 Top Solution [Top 5% Solution](https://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge/discussion/5775),[PDF](https://kaggle2.blob.core.windows.net/forum-message-attachments/9133/PZAD_slides_sticky%20notes%20translation.pdf)

Key Methods: Feature Selection

### 6.4 Comment

solution set不是很丰富，披着nlp的外衣，其实还是在抽象的feature做选择，**高频交易**