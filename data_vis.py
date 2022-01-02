import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt
series1 = pd.read_csv('dataset1.csv', header=0, parse_dates=[0])
#pyplot.plot(series1)
series2 = pd.read_csv('dataset2.csv', header=0, parse_dates=[0])
ndata = series1[ (series1['TIME'] > '2020-01-01') & (series1['TIME'] < '2020-04-01')] 
#pyplot.plot(series1)
#pyplot.plot(series1)

#series1.plot(x ='TIME', y='AVAILABLE BIKES')
#series2.plot(x ='TIME', y='AVAILABLE BIKES')
ndata.plot(x ='TIME', y='AVAILABLE BIKES')
#ndata.plot(x ='TIME', y='AVAILABLE BIKES')

# from statsmodels.tsa.stattools import adfuller 

# result = adfuller(series1['AVAILABLE BIKES'])
# print('ADF Statistic[Heuston Station]: %f' % result[0])
# print('p-value[Heuston Station]: ', result[1])
# print('Critical Values[Heuston Station]:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))
# print('\n')
# result = adfuller(series2['AVAILABLE BIKES'])
# print('ADF Statistic[Portobello Harbour]: %f' % result[0])
# print('p-value[Portobello Harbour]: ', result[1])
# print('Critical Values[Portobello Harbour]:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))


# from statsmodels.tsa.seasonal import seasonal_decompose

# series1.set_index('TIME', inplace=True)
# analysis = ndata[['AVAILABLE BIKES']].copy()
# decompose_result_mult = seasonal_decompose(analysis, model="additive", period=288)
# trend = decompose_result_mult.trend
# seasonal = decompose_result_mult.seasonal
# residual = decompose_result_mult.resid

# decompose_result_mult.plot()
# plt.show()

df = pd.read_csv('dataset1.csv', usecols = [1,6], parse_dates=[1])

# 3rd Feb 2020 is a monday, 10th is following monday
start=pd.to_datetime("04-02-2020",format='%d-%m-%Y')
end=pd.to_datetime("14-03-2020",format='%d-%m-%Y')
# convert date/time to unix timestamp in sec
t_full=pd.array(pd.DatetimeIndex(df.iloc[:,0]).astype(np.int64))/1000000000
dt = t_full[1]-t_full[0]
print("data sampling interval is %d secs"%dt)

# # extract data between start and end dates
t_start = pd.DatetimeIndex([start]).astype(np.int64)/1000000000
t_end = pd.DatetimeIndex([end]).astype(np.int64)/1000000000
# print(len(t_start))
# print(len(t_end))
# print(len(t_full))
#t = np.extract([(t_full>=t_start) & (t_full<=t_end)], t_full)
t = series1[ (series1['TIME'] > '2020-02-04') & (series1['TIME'] < '2020-03-14')] 

#t=(t-t[0])/60/60/24 # convert timestamp to days
#y = np.extract([(t_full>=t_start) & (t_full<=t_end)], df.iloc[:,1]).astype(np.int64)
#y = series1[ (series1['AVAILABLE BIKES'] > '2020-02-04') & (series1['AVAILABLE BIKES'] < '2020-03-14')] 
y = t['AVAILABLE BIKES']
# t=series1['TIME']
# y=series1['AVAILABLE BIKES']
# #plot extracted data
# plt.scatter(t,y, color='red', marker='.'); plt.show()


def test_preds(q,dd,lag,plot):
    print(test_preds)
    #q−step ahead prediction
    stride=1
    XX=y[0:y.size-q-lag*dd:stride]
    for i in range(1,lag):
        X=y[i*dd:y.size-q-(lag-i)*dd:stride]
        XX=np.column_stack((XX,X))
    yy=y[lag*dd+q::stride]; tt=t[lag*dd+q::stride]
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)
    from sklearn.linear_model import Ridge
    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)
    if plot:
        y_pred = model.predict(XX)
        plt.scatter(t, y, color='black'); plt.scatter(tt, y_pred, color='blue')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions"],loc='upper right')
        day=math.floor(24*60*60/dt) # number of samples per day
        plt.xlim(((lag*dd+q)/day,(lag*dd+q)/day+2))
        plt.show()


# prediction using short−term trend
plot=True
test_preds(q=10,dd=1,lag=3,plot=plot)
































