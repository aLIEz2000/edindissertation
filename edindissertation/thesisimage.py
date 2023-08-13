import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.projections import get_projection_class

'''
# Data for window, horizon, and MASE values
window = [4, 6, 8]
horizon = [1, 2, 3]
models = ['Simple RNN', 'CRNN', 'LSTM']

# MASE values for each model and training period
first_training_mase = [
    [2.416122875, 1.794367213, 1.951863094],
    [2.492825676, 2.014192271, 2.805807868],
    [2.75974801, 2.114180449, 2.438567847]
]

second_training_mase = [
    [1.626038234, 1.250919552, 1.377252248],
    [1.674547247, 1.197435441, 1.830667774],
    [2.298029704, 1.679453704, 1.94963202]
]

third_training_mase = [
    [2.003464309, 1.55329881, 1.830687279],
    [2.469063809, 1.578148461, 1.908004222],
    [2.764673685, 1.766990171, 1.802380289]
]

fourth_training_mase = [
    [1.684357353, 1.031510755, 1.236976143],
    [1.501602153, 1.1435323, 1.121746579],
    [1.721282289, 0.953425545, 1.212160322]
]

# Grouped Bar Chart
bar_width = 0.2
index = np.arange(len(window))
fig, ax = plt.subplots()

for i, model in enumerate(models):
    first = ax.bar(index - 1.5 * bar_width + i * bar_width, first_training_mase[i], bar_width, label=model + " (1st)")
    second = ax.bar(index - 0.5 * bar_width + i * bar_width, second_training_mase[i], bar_width, label=model + " (2nd)")
    third = ax.bar(index + 0.5 * bar_width + i * bar_width, third_training_mase[i], bar_width, label=model + " (3rd)")
    fourth = ax.bar(index + 1.5 * bar_width + i * bar_width, fourth_training_mase[i], bar_width, label=model + " (4th)")

ax.set_xlabel('Window and Horizon')
ax.set_ylabel('MASE Values')
ax.set_title('Comparison of MASE Values for Different Models and Training Periods')
ax.set_xticks(index)
ax.set_xticklabels([f'{w},{h}' for w, h in zip(window, horizon)])
ax.legend()

plt.show()


# Line Chart
fig, ax = plt.subplots()

for i, model in enumerate(models):
    ax.plot(horizon, first_training_mase[i], label=model + " (1st)")
    ax.plot(horizon, second_training_mase[i], label=model + " (2nd)")
    ax.plot(horizon, third_training_mase[i], label=model + " (3rd)")
    ax.plot(horizon, fourth_training_mase[i], label=model + " (4th)")

ax.set_xlabel('Horizon')
ax.set_ylabel('MASE Values')
ax.set_title('Comparison of MASE Values for Different Models and Training Periods')
ax.legend()

plt.show()



# Heatmap
fig, ax = plt.subplots()

im = ax.imshow(first_training_mase, cmap='YlGn')
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(window)))

ax.set_xticklabels(models)
ax.set_yticklabels(window)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(window)):
    for j in range(len(models)):
        text = ax.text(j, i, f'{first_training_mase[i][j]:.2f}',
                       ha="center", va="center", color="black")

ax.set_xlabel('Models')
ax.set_ylabel('Window')
ax.set_title('Heatmap of MASE Values for the First Training Period')
plt.colorbar(im)

plt.show()



# Parallel Coordinates Chart
fig, ax = plt.subplots()
parallel_coordinates_data = [first_training_mase, second_training_mase, third_training_mase, fourth_training_mase]

for i, model in enumerate(models):
    ax.plot(range(len(horizon)), parallel_coordinates_data[i], label=model)

ax.set_xticks(range(len(horizon)))
ax.set_xticklabels(horizon)
ax.set_xlabel('Horizon')
ax.set_ylabel('MASE Values')
ax.set_title('Parallel Coordinates Chart of MASE Values for Different Models and Training Periods')
ax.legend()

plt.show()



# Radar Chart
categories = [f'({w},{h})' for w in window for h in horizon]

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

for i, model in enumerate(models):
    values = np.concatenate(parallel_coordinates_data[i])
    ax.plot(np.linspace(0, 2 * np.pi, len(values), endpoint=False), values, label=model)

ax.set_xticks(np.linspace(0, 2 * np.pi, len(values), endpoint=False))
ax.set_xticklabels(categories)
ax.set_title('Radar Chart of MASE Values for Different Models and Training Periods')
ax.legend()

plt.show()
'''


'''
# Data
Period = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]
Window = [4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8, 4, 6, 8]
Horizon = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3]
Simple_RNN = [2.416122875, 2.492825676, 2.75974801, 1.87756535, 2.585855942, 2.705735656, 1.799490233, 2.133598282, 2.646420137, 1.626038234, 1.674547247, 2.298029704, 1.489319956, 1.865809993, 2.279945623, 1.232437544, 1.6209055, 2.120110437, 2.003464309, 2.469063809, 2.764673685, 1.681211178, 2.089980514, 2.284850781, 1.22601363, 1.855671421, 2.260419521, 1.684357353, 1.501602153, 1.721282289, 1.265724333, 1.317020633, 1.254744657, 0.85469817, 1.104479577, 1.074261898]
CRNN = [1.794367213, 2.014192271, 2.114180449, 1.38102012, 1.787297325, 1.732811236, 1.479058097, 1.440710071, 1.531138152, 1.250919552, 1.197435441, 1.679453704, 1.312237414, 1.029185564, 1.460181058, 0.862000424, 0.848976875, 1.579699537, 1.55329881, 1.578148461, 1.766990171, 1.316504556, 1.184598459, 1.348768216, 1.018471386, 1.095755723, 1.395615081, 1.031510755, 1.1435323, 0.953425545, 0.698735692, 0.803789033, 0.810007613, 0.707213785, 0.65853, 0.726772559]
LSTM = [1.951863094, 2.805807868, 2.438567847, 1.60585206, 2.335023986, 2.47825206, 1.408347041, 1.627827224, 1.74617763, 1.377252248, 1.830667774, 1.94963202, 1.17721491, 1.281387361, 1.559422695, 1.130633724, 1.482448839, 1.087115701, 1.830687279, 1.908004222, 1.802380289, 1.304700218, 1.95030624, 2.10444236, 1.046436907, 1.234286759, 1.327360772, 1.236976143, 1.121746579, 1.212160322, 0.749471046, 0.827724195, 1.061060226, 0.60889706, 0.61936982, 0.982726997]

# Create 3D scatter plot for each model
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot data points for each model
ax.scatter(Period, Window, Horizon, c=Simple_RNN, marker='o', label='Simple RNN')
ax.scatter(Period, Window, Horizon, c=CRNN, marker='^', label='CRNN')
ax.scatter(Period, Window, Horizon, c=LSTM, marker='s', label='LSTM')

ax.set_xlabel('Period')
ax.set_ylabel('Window')
ax.set_zlabel('Horizon')
ax.set_title('Performance of Models')
ax.legend()

plt.show()
'''




'''
df = pd.read_excel('C:/Users/patrick/Desktop/result3.xlsx', engine='openpyxl')

windows = sorted(df['Window'].unique())
horizons = sorted(df['Horizon'].unique())
feature_dimensions = sorted(df['Period'].unique())


window_vals = sorted(df['Window'].unique())
horizon_vals = sorted(df['Horizon'].unique())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), subplot_kw={'projection': '3d'})

# first subplot
ax1 = axes[0]

for p in Period:
    subset = df[df['Period'] == p]
    ax1.scatter(subset['Window'], subset['Horizon'], subset['MSE'], label=f'Period={f}')

ax1.set_xlabel('Window')
ax1.set_ylabel('Horizon')
ax1.set_zlabel('MSE')
ax1.set_zlim(0, 0.00008)
ax1.set_xticks(window_vals)
ax1.set_yticks(horizon_vals)
ax1.legend()

# second subplot
ax2 = axes[1]

# Create a grid dataset
X, Y = np.meshgrid(window_vals, horizon_vals)

# Calculate the MSE value for each point
Z = []
for h in horizon_vals:
    mse_vals = []
    for w in window_vals:
        mse_list = [df.loc[(df['Window'] == w) & (df['Horizon'] == h) & (df['Feature Dimension'] == f), 'MSE'].values[0] for f in feature_dimensions]
        mse_vals.append(mse_list)
    Z.append(mse_vals)
Z = np.array(Z)

# draw surface
for idx, f in enumerate(feature_dimensions):
    ax2.plot_surface(X, Y, Z[:, :, idx], alpha=0.7, cmap='rainbow', label=f'Feature Dimension={f}')

ax2.set_xlabel('Window', fontsize=10)
ax2.set_ylabel('Horizon', fontsize=10)
ax2.set_zlabel('MSE', fontsize=10)
ax2.set_zlim(0, 0.00008)
ax2.set_xticks(window_vals)
ax2.set_yticks(horizon_vals)

plt.show()



# Convert data to a format suitable for heatmaps
heatmap_data = df.pivot_table(index=['Window', 'Feature Dimension'], columns='Horizon', values='MSE')

# Create a heatmap
plt.figure(figsize=(12, 8))

# Set the range of the colormap
vmin = heatmap_data.min().min()
vmax = heatmap_data.max().max() * 0.0005  # You can adjust this value to increase or decrease the color gap

sns.heatmap(heatmap_data, annot=True, fmt='.2e', cmap='magma', linewidths=0.5, annot_kws={'fontsize': 10}, vmin=vmin, vmax=vmax)
plt.title('MSE for Different Parameter Combinations', fontsize=16)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('(Window, Feature Dimension)', fontsize=14)

# Show heatmap
plt.show()
'''

'''
# Data for each method's best counts
methods = ['Simple RNN', 'CRNN', 'LSTM']
best_counts = [0, 28, 8]
colors = ['#FF66CC', '#FF9900', '#993399']  # Pink, Orange, Purple

# Create a bar chart to show the number of times each method is the best
plt.figure(figsize=(8, 6))
plt.bar(methods, best_counts, color=colors, edgecolor='black', linewidth=1.2)
plt.title("Number of Times Each Method is the Best", fontsize=16)
plt.xlabel("Method", fontsize=14)
plt.ylabel("Number of Times", fontsize=14)

# Customize tick labels and grid
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate the bars with the counts
for i, count in enumerate(best_counts):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Show the plot
plt.tight_layout()
plt.show()
'''


'''
# Generate some sample data for the two forecasts
days = [1, 2, 3, 4, 5]
fluctuating_forecast = [100, 105, 98, 110, 100]  # Sample fluctuating forecast
non_fluctuating_forecast = [100, 100, 100, 100, 100]  # Sample non-fluctuating forecast

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the fluctuating forecast with a wavy pattern
ax.plot(days, fluctuating_forecast, label='Fluctuating Forecast (LSTM)', color='#FF9900', linestyle='-', linewidth=2)
ax.fill_between(days, fluctuating_forecast, alpha=0.5, color='#FF9900')

# Plot the non-fluctuating forecast with a straight line
ax.plot(days, non_fluctuating_forecast, label='Non-Fluctuating Forecast (CRNN)', color='#993399', linestyle='-', linewidth=2)

# Add labels, title, and legend
ax.set_xlabel('Days in the Future', fontsize=14)
ax.set_ylabel('Forecasted Price', fontsize=14)
ax.set_title('Contrasting Forecast Behavior', fontsize=16, fontweight='bold')

# Set y-axis starting point to 80
ax.set_ylim(80, max(max(fluctuating_forecast), max(non_fluctuating_forecast)) + 5)

# Customize the x-axis ticks
ax.set_xticks(days)
ax.set_xticklabels(days)

# Customize the plot background and grid
ax.set_facecolor('#F7F7F7')
ax.grid(True, linestyle='--', alpha=0.7)

# Add annotations to highlight the difference
ax.annotate('Fluctuating Forecast', xy=(2.5, 105), xytext=(2.8, 108), arrowprops=dict(arrowstyle="->", color='#FF9900'), fontsize=12)
ax.annotate('Non-Fluctuating Forecast', xy=(2.5, 100), xytext=(2.8, 97), arrowprops=dict(arrowstyle="->", color='#993399'), fontsize=12)

# Add a creative subtitle
plt.text(2.5, max(max(fluctuating_forecast), max(non_fluctuating_forecast)) + 8, 'Visualizing the Dynamics of Forecasted Prices', fontsize=14, color='#4F4F4F')

# Remove spines and legends
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
'''


'''
# Read the data from the Excel file
df = pd.read_excel('C:/Users/patrick/Desktop/result3.xlsx', engine='openpyxl')

# Group the data by period
grouped = df.groupby('Period')

# Define method names and colors for plotting
methods = ['Simple RNN', 'CRNN', 'LSTM']
colors = ['b', 'g', 'r']

# Create a 2x2 subplot for 3D plots
fig = plt.figure(figsize=(14, 10))

# Iterate over each period and plot the 3D scatter plot in the corresponding subplot
for i, (period, data) in enumerate(grouped):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    if period == 1:
        ax.set_title("First Training")
    elif period == 2:
        ax.set_title("Second Training")
    elif period == 3:
        ax.set_title("Third Training")
    elif period == 4:
        ax.set_title("Fourth Training")
    ax.set_xlabel('Window')
    ax.set_ylabel('Horizon')
    ax.set_zlabel('MASE')

    # Get unique window and horizon combinations for each period
    unique_windows = data['Window'].unique()
    unique_horizons = data['Horizon'].unique()

    # Generate grid for 3D plotting
    X, Y = np.meshgrid(unique_windows, unique_horizons)
    Z = np.zeros(X.shape)

    # Plot the 3D scatter plot for each method
    for j, method in enumerate(methods):
        for k, window in enumerate(unique_windows):
            for l, horizon in enumerate(unique_horizons):
                mase_value = data.loc[(data['Window'] == window) & (data['Horizon'] == horizon), method].iloc[0]
                Z[l, k] = mase_value

        ax.scatter(X, Y, Z, c=colors[j], label=method)

    # Set the legend outside the plot to avoid overlapping
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Adjust spacing between subplots
plt.tight_layout()

# Show the combined plot
plt.show()
'''


'''
# Read the data from the Excel file
df = pd.read_excel('C:/Users/patrick/Desktop/result3.xlsx', engine='openpyxl')

# Group the data by period
grouped = df.groupby('Period')

# Define method names and colors for plotting
methods = ['Simple RNN', 'CRNN', 'LSTM']
colors = ['pink', 'orange', 'purple']

# Create four 3D plots, one for each period
fig = plt.figure(figsize=(16, 12))

# Titles for each training
titles = ["First Training", "Second Training", "Third Training", "Fourth Training"]

for idx, (period, data) in enumerate(grouped, 1):
    ax = fig.add_subplot(2, 2, idx, projection='3d')
    ax.set_title(titles[idx-1])  # Set the title for each subplot
    ax.set_xlabel('Window')
    ax.set_ylabel('Horizon')
    ax.set_zlabel('MASE')

    # Get unique window and horizon combinations for each period
    unique_windows = data['Window'].unique()
    unique_horizons = data['Horizon'].unique()

    # Generate grid for 3D plotting
    X, Y = np.meshgrid(unique_windows, unique_horizons)

    # Create array to store MASE values for each method
    MASE_values = np.zeros((len(unique_windows), len(unique_horizons), len(methods)))

    # Store the MASE values for each method in the array
    for i, method in enumerate(methods):
        for j, window in enumerate(unique_windows):
            for k, horizon in enumerate(unique_horizons):
                mase_value = data.loc[(data['Window'] == window) & (data['Horizon'] == horizon), method].iloc[0]
                MASE_values[j, k, i] = mase_value

        # Plot the 3D surface plot for each method
        ax.plot_surface(X, Y, MASE_values[:, :, i], color=colors[i], alpha=0.7)

    # Create custom legend outside the plot
    legend_labels = methods
    proj3d = get_projection_class('3d')(fig)
    handles = [plt.Line2D([], [], linestyle='', marker='o', markersize=10, color=colors[i]) for i in range(len(methods))]
    legend = plt.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    ax.add_artist(legend)

# Adjust layout and spacing between subplots
plt.tight_layout()

# Show the plot with all four subplots
plt.show()
'''


'''
# Read the data from the Excel file
df = pd.read_excel('C:/Users/patrick/Desktop/result3.xlsx', engine='openpyxl')

# Create a new column to combine window and horizon
df['Window-Horizon'] = df['Window'].astype(str) + '-' + df['Horizon'].astype(str)

# Select the relevant columns for CRNN and LSTM
crnn_data = df[['Period', 'Window-Horizon', 'CRNN']]
lstm_data = df[['Period', 'Window-Horizon', 'LSTM']]

# Pivot the data to create the heatmap format
crnn_heatmap_data = crnn_data.pivot(index='Window-Horizon', columns='Period', values='CRNN')
lstm_heatmap_data = lstm_data.pivot(index='Window-Horizon', columns='Period', values='LSTM')

# Create the CRNN heatmap
plt.figure(figsize=(10, 6))
plt.subplot(121)
sns.heatmap(crnn_heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
plt.title('CRNN - MASE Heatmap')
plt.xlabel('Period')
plt.ylabel('Window-Horizon')
plt.xticks(rotation=45)

# Create the LSTM heatmap
plt.subplot(122)
sns.heatmap(lstm_heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
plt.title('LSTM - MASE Heatmap')
plt.xlabel('Period')
plt.ylabel('Window-Horizon')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
'''




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
sequence_length = 4

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

    for start, stop in zip(range(seq_length-2, num_elements-pred_length+1), range(seq_length+pred_length-2, num_elements+1)):
        yield data_matrix[start:stop, :]

pred_length = 3
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



'''
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

model.save('lstm_model.h5')
result = model.predict(X_test, batch_size=100, verbose=0)


overall_mae = []
overall_mse = []
overall_mape = []
overall_mase = []

len_test = len(y_test)
print("测试集的长度:", len(y_test))

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
'''

'''

# Randomly choose three companies
unique_companies = df_sorted['Name'].unique()
random_companies = random.sample(list(unique_companies), 3)
print("Randomly selected companies:", random_companies)

# Iterate through each randomly selected company
for random_company in random_companies:
    # Prepare input data for the selected company
    company_data = df[df['Name'] == random_company]
    company_test_data = company_data[company_data['test_flag'] == True]

    # Drop the initial rows to match the sequence length
    plot_data = company_test_data.drop(company_test_data.head(sequence_length).index)

    # Construct input features
    X_random_company = []

    for seq in gen_sequence(company_test_data, sequence_length, col):
        X_random_company.append(seq)
    X_random_company = np.asarray(X_random_company)

    # Normalize input data
    X_random_company = scaler.transform(X_random_company.reshape(-1, X_random_company.shape[-1])).reshape(X_random_company.shape)

    # Use the trained model to make predictions
    model = load_model('lstm_model.h5')
    predicted_price = model.predict(X_random_company, batch_size=100, verbose=0)
    Pre_price = np.reshape(predicted_price, (-1, 1))
    y_predict = pd.DataFrame(Pre_price, columns=['Adj_Close'])

    # Plot predictions and true values on the same graph
    plt.figure(figsize=(12, 6))
    plt.plot(company_test_data.index[sequence_length:sequence_length + 34], y_predict['Adj_Close'][:34],label='Predicted Price', color='orange')
    plt.plot(company_test_data.index[sequence_length:], plot_data['Adj_Close'], label='Actual Price', color='purple')
    plt.ylim(0, y_predict['Adj_Close'][:34].max() * 1.2)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{random_company} Price Prediction vs Actual')
    plt.legend()
    plt.show()
    '''

'''
# Read the data from the Excel file
df = pd.read_excel('C:/Users/patrick/Desktop/result3.xlsx', engine='openpyxl')

# Create a new column to combine window and horizon
df['Window-Horizon'] = df['Window'].astype(str) + '-' + df['Horizon'].astype(str)

# Select the relevant columns for CRNN and LSTM
crnn_data = df[['Period', 'Window-Horizon', 'CRNN']]
lstm_data = df[['Period', 'Window-Horizon', 'LSTM']]

# Pivot the data to create the heatmap format
crnn_heatmap_data = crnn_data.pivot(index='Window-Horizon', columns='Period', values='CRNN')
lstm_heatmap_data = lstm_data.pivot(index='Window-Horizon', columns='Period', values='LSTM')

# Create the CRNN heatmap with annotations and contour lines
plt.figure(figsize=(12, 6))
plt.subplot(121)
sns.heatmap(crnn_heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
plt.title('CRNN - MASE Heatmap')
plt.xlabel('Period')
plt.ylabel('Window-Horizon')
plt.xticks(rotation=45)
plt.contour(crnn_heatmap_data, colors='k', linestyles='dashed', levels=10)  # Add contour lines

# Create the LSTM heatmap with annotations and contour lines
plt.subplot(122)
sns.heatmap(lstm_heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
plt.title('LSTM - MASE Heatmap')
plt.xlabel('Period')
plt.ylabel('Window-Horizon')
plt.xticks(rotation=45)
plt.contour(lstm_heatmap_data, colors='k', linestyles='dashed', levels=10)  # Add contour lines

plt.tight_layout()
plt.show()
'''












