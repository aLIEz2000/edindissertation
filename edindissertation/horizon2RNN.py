# python -m pip install --upgrade pip


import os
from os import listdir
import random
import tqdm
import numpy as np
import pandas as pd
import time
import csv

import tensorflow as tf
from tensorflow.python import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import backend as K
from keras.layers import Dense, Embedding, SimpleRNN
from keras.models import Sequential
from scipy import stats

from pprint import pprint
import matplotlib
from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.options.display.precision=4
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import stats
import scipy.cluster.hierarchy as sch

import yfinance as yf



start_time = time.time()

# remove an useless column in raw dataset
df_sorted = pd.read_csv('D:/python/DATA/thesis/Training/fourth_training.csv')
df = df_sorted.drop(['Volume', 'Daily_Returns'], axis=1)
print(df.head(10))

# construct a new column with tag of train or test


# total days in dataset
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df.sort_values('Date',inplace=True)
df.set_index('Date',inplace=True)


df.info()
print(df.info())

Time_diff = df_sorted['Date'].nunique()
print(Time_diff, 'days in the dataset')
num_companies = df_sorted['Name'].nunique()
print(num_companies, 'stocks in total in the datasets')




aapl = df[df['Name'] == 'AAPL']

aapl.to_csv('D:/python/DATA/thesis/Training/aapl.csv', index=True)



train_start = aapl.index.get_loc(pd.to_datetime('31/03/2021', format="%d/%m/%Y"))
print(train_start)
train_end = aapl.index.get_loc(pd.to_datetime('31/03/2023', format="%d/%m/%Y"))
print(train_end)
test_start = aapl.index.get_loc(pd.to_datetime('03/04/2023', format="%d/%m/%Y"))
print(test_start)
test_end = aapl.index.get_loc(pd.to_datetime('26/05/2023', format="%d/%m/%Y"))
print(test_end)




# create a new column of test flag

for company in df['Name'].unique():
    company_indices = df.loc[df['Name'] == company].index
    df.loc[company_indices[train_start:train_end], 'test_flag'] = False
    df.loc[company_indices[test_start:test_end], 'test_flag'] = True

df.sort_values(['Date', 'Name'], inplace=True)




### CREATE GENERATOR FOR LSTM WINDOWS AND LABELS ###

# the length of each training sample
# need to chage sequence_length manually
sequence_length = 8

def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

# zip takes an iterable (list, tuple, set, or dictionary), generates a list of tuples that contain elements from each iterable
# construct LSTM input features
def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


# construct LSTM output
def gen_labels(id_df, seq_length, label, pred_length):

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(seq_length-1, num_elements-pred_length+1), range(seq_length+pred_length-1, num_elements+1)):
        yield data_matrix[start:stop, :]

pred_length = 2
# need to chage pred_length manually

### CREATE TRAIN/TEST PRICE DATA ###

# create a list of columns to exclude from the sequence
exclude_cols = ['Name', 'test_flag']

# get a list of columns to include in the sequence
col = [col for col in df.columns if col not in exclude_cols]

X_train, X_test = [], []
y_train, y_test = [], []

for (stock, is_test), _df in df.groupby(['Name', 'test_flag']):

    for seq in gen_sequence(_df, sequence_length, col):
        if is_test:
            X_test.append(seq)
        else:
            X_train.append(seq)

    for seq in gen_labels(_df, sequence_length, ['Adj_Close'], pred_length):
            if is_test:
                y_test.append(seq)
            else:
                y_train.append(seq)



X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print('shape of the intput X and output y\ninput training data dimension: ')
print(X_train.shape)
print('\noutput training data dimension: ')
print(y_train.shape)

print('\ninput testing data dimension: ')
print(X_test.shape)
print('\noutput testing data dimension: ')
print(y_test.shape)

# print several examples, print all features
for i in range(2):
    print('input ', i, ': \n', X_train[i, :, 0:8])
    print('==> output ', i, ': \n', y_train[i], '\n')





scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1,X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1,X_test.shape[-1])).reshape(X_test.shape)

print(X_train.shape)

# shape of the training data is (3367130, 4, 7), it means:
# 3367130 rows
# 4 rows --> 1 output (LSTM many to many application)
# 7 features in each row



# construct model


# Simple RNN
set_seed(33)

inputs = Input(shape=(X_train.shape[1:]))
enc = SimpleRNN(128, activation='relu', return_sequences=False)(inputs)
x = RepeatVector(pred_length)(enc)
dec = SimpleRNN(32, activation='relu', return_sequences=True)(x)
out = TimeDistributed(Dense(1))(dec)

model = Model(inputs, out)
model.compile(loss='mse', optimizer='adam')

# 训练模型
es = EarlyStopping(patience=6, verbose=2, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_train, y_train),
          epochs=50, batch_size=100, verbose=1, callbacks=[es])

# 预测结果
result = model.predict(X_test, batch_size=100, verbose=0)

# 计算指标
overall_mae = []
overall_mse = []
overall_mape = []
overall_mase = []

len_test = len(y_test)
print("测试集的长度:", len(y_test))

for i in range(2):
    print('Forecasted values:')
    print(result[i, :, :])
    print('True values:')
    print(y_test[i, :, :], '\n\n')

# CALCULATE METRICS
for i in range(len_test):
    overall_mae.append(mean_absolute_error(result[i], y_test[i]))
    overall_mse.append(mean_squared_error(result[i], y_test[i]))
    overall_mape.append(mean_absolute_percentage_error(result[i], y_test[i]))
    overall_mase.append(mean_absolute_error(result[i], y_test[i]) / mean_absolute_error(np.squeeze(y_train[1:]), np.squeeze(y_train[:-1])))

# 打印整体指标
print('Simple RNN 预测的整体 MAE:', np.mean(overall_mae))
print('Simple RNN 预测的整体 MSE:', np.mean(overall_mse))
print('Simple RNN 预测的整体 MAPE:', np.mean(overall_mape))
print('Simple RNN 预测的整体 MASE:', np.mean(overall_mase))

end_time = time.time()

execution_time = end_time - start_time
print("执行时间:", execution_time/60, '分钟')



# CRNN (Convolutional Recurrent Neural Network)
set_seed(33)
# Input shape: (sequence_length, num_features)

inputs = Input(shape=(X_train.shape[1:]))

# Convolutional layers
x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)

# Recurrent layers
x = LSTM(128, activation='relu', return_sequences=True)(x)
x = LSTM(32, activation='relu', return_sequences=True)(x)
x = RepeatVector(pred_length)(x[:, -1, :])

# Output layer
out = TimeDistributed(Dense(1))(x)

model = Model(inputs, out)
model.compile(loss='mse', optimizer='adam')

### TRAIN MODEL ###
es = EarlyStopping(patience=6, verbose=2, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_train, y_train),
          epochs=50, batch_size=100, verbose=1, callbacks=[es])

### MAKE PREDICTIONS ###
result = model.predict(X_test, batch_size=100, verbose=0)



### CALCULATE METRICS ###
overall_mae = []
overall_mse = []
overall_mape = []
overall_mase = []



print("测试集的长度:", len(y_test))

for i in range(2):
    print('Forecasted values:')
    print(result[i, :, :])
    print('True values:')
    print(y_test[i, :, :], '\n\n')

# CALCULATE METRICS
for i in range(len_test):
    overall_mae.append(mean_absolute_error(result[i], y_test[i]))
    overall_mse.append(mean_squared_error(result[i], y_test[i]))
    overall_mape.append(mean_absolute_percentage_error(result[i], y_test[i]))
    overall_mase.append(mean_absolute_error(result[i], y_test[i]) / mean_absolute_error(np.squeeze(y_train[1:]), np.squeeze(y_train[:-1])))




### PRINT METRICS ###
print('Overall MAE of CRNN forecasting:', np.mean(overall_mae))
print('Overall MSE of CRNN forecasting:', np.mean(overall_mse))
print('Overall MAPE of CRNN forecasting:', np.mean(overall_mape))
print('Overall MASE of CRNN forecasting:', np.mean(overall_mase))


end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time/60, 'minutes')






### Using Simple LSTM model

set_seed(33)

### DEFINE FORECASTER ###

inputs = Input(shape=(X_train.shape[1:]))
enc = LSTM(128, activation='relu', return_sequences=False)(inputs)
x   = RepeatVector(pred_length)(enc)
dec = LSTM(32, activation='relu', return_sequences=True)(x)
out = TimeDistributed(Dense(1))(dec)

model = Model(inputs, out)
model.compile(loss='mse', optimizer='adam')

### FIT FORECASTER ###
es = EarlyStopping(patience=6, verbose=2, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_train, y_train),
          epochs=50, batch_size=100, verbose=1, callbacks=[es])


result = model.predict(X_test, batch_size=100, verbose=0)


overall_mae = []
overall_mse = []
overall_mape = []
overall_mase = []

print("测试集的长度:", len(y_test))

for i in range(2):
    print('Forecasted values:')
    print(result[i, :, :])
    print('True values:')
    print(y_test[i, :, :], '\n\n')

# CALCULATE METRICS
for i in range(len_test):
    overall_mae.append(mean_absolute_error(result[i], y_test[i]))
    overall_mse.append(mean_squared_error(result[i], y_test[i]))
    overall_mape.append(mean_absolute_percentage_error(result[i], y_test[i]))
    overall_mase.append(mean_absolute_error(result[i], y_test[i]) / mean_absolute_error(np.squeeze(y_train[1:]), np.squeeze(y_train[:-1])))

# Print out overall metrics
print('Overall MAE of LSTM forecasting:', np.mean(overall_mae))
print('Overall MSE of LSTM forecasting:', np.mean(overall_mse))
print('Overall MAPE of LSTM forecasting:', np.mean(overall_mape))
print('Overall MASE of LSTM forecasting:', np.mean(overall_mase))


end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time/60, 'minutes')








