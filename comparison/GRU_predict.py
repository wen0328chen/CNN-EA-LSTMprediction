import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import datetime
import tensorflow
import tensorflow as tf
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, Sequential, callbacks
from tensorflow.keras.layers import  Flatten, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import SGD, Adam
import math
from sklearn.metrics import mean_squared_error
import pickle

# print(tf.config.list_physical_devices('GPU'))

df = pd.read_csv('./dataset/LSTM-Multivariate_pollution.csv')
# print(df.head())

df['date']=pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['date_num'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day_name()
df.drop('date',axis=1,inplace=True)
# print(df.head())


# Replace object values in 'wnd_dir' with numerical values
wnd_dir_mapping = {
    'NE': 0,
    'NW': 1,
    'SE': 2,
    'cv': 3  # Add other directions as needed
}
df['wnd_dir'] = df['wnd_dir'].replace(wnd_dir_mapping)

# drop day column with object dtype
df.drop('day', axis=1, inplace=True)

# MinMaxScale all the feature columns also for pollution
# Create a MinMaxScaler object
scaler = MinMaxScaler()
# Fit and transform the selected columns
columns_to_scale = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(df.head())

# be careful of fitting before
scaler_y = MinMaxScaler()
columns_y = ['pollution']
df[columns_y] = scaler_y.fit_transform(df[columns_y])
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# scaler_features = MinMaxScaler()
# # Fit and transform the selected columns
# columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# df[columns] = scaler_features.fit_transform(df[columns])
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler_features, f)


# df.to_csv('./dataset/data_processed.csv', index=False, encoding='utf-8')


# create sequence with target prediction for pollution
def create_sequences(data, seq_length, target_col):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length), 0:data.shape[1]]
        y = data[i+seq_length, target_col]  # Target is pollution
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
seq_length = 5
target_column_index = 0 # Index of the 'pollution' column
X, y = create_sequences(df.values, seq_length, target_column_index)
# split into train and test sets:
train_size = int(len(X) * 0.8)  # Example: 80% for training
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]
print("Shape of training data X:", X_train.shape)
print("Shape of training data y:", y_train.shape)
print("Shape of testing data X:", X_test.shape)
print("Shape of testing data y:", y_test.shape)


#%%
# Build the GRU model
from tensorflow.keras.layers import GRU
model = Sequential()
model.add(GRU(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))  # Early Dropping
# model.add(GRU(units=64, return_sequences=True)) #!!
# model.add(Dropout(0.1))
model.add(GRU(units=32))
model.add(Dropout(0.1))
# model.add(Dense(units=16, activation='relu')) #!!
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
model.summary()


# Define early stopping
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(filepath='GRU_best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
# Train the model with early stopping
with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data= (X_test,y_test), callbacks=[early_stopping, checkpoint])


# Load the best model
best_model = load_model('GRU_best_model.h5')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)


# Reshape y_pred to have the same number of features the scaler was trained on
y_pred_reshaped = y_pred.reshape(-1, 1)
# Concatenate with zeros to match the original number of features before scaling
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred_reshaped, np.zeros((len(y_pred_reshaped), len(columns_to_scale) - 1))], axis=1))[:, 0]
# Reshape y_test to have the same number of features the scaler was trained on
y_test_reshaped = y_test.reshape(-1, 1)
# Concatenate with zeros to match the original number of features before scaling
y_test_rescaled = scaler.inverse_transform(np.concatenate([y_test_reshaped, np.zeros((len(y_test_reshaped), len(columns_to_scale) - 1))], axis=1))[:, 0]
# Now  use y_pred_rescaled and y_test_rescaled for evaluation
rmse = math.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print('RMSE (Rescaled):', rmse)

r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print('R-squared:', r2)

#  plot test vs predicted for last 1 month
last_month_data = df.iloc[-744:]
# Get corresponding predictions and actual values
last_month_predictions = y_pred_rescaled[-744:]
last_month_actual = y_test_rescaled[-744:]
plt.figure(figsize=(12, 6))
plt.plot(last_month_actual, label='Actual')
plt.plot(last_month_predictions, label='Predicted')
plt.xlabel('Time Steps (last month)')
plt.ylabel('Pollution')
plt.title('Actual vs Predicted Pollution (Last Month)')
plt.legend()
# plt.show()

#  plot test vs predicted for last week
# Get the last week's data (assuming your data is hourly)
last_week_data = df.iloc[-168:]  # Last 168 hours (24 hours/day * 7 days)
# Get corresponding predictions and actual values
last_week_predictions = y_pred_rescaled[-168:]
last_week_actual = y_test_rescaled[-168:]
plt.figure(figsize=(12, 6))
plt.plot(last_week_actual, label='Actual')
plt.plot(last_week_predictions, label='Predicted')
plt.xlabel('Time Steps (last week)')
plt.ylabel('Pollution')
plt.title('Actual vs Predicted Pollution (Last Week)')
plt.legend()
# plt.show()


#  plot anomalies
# Calculate the absolute difference between actual and predicted values
errors = np.abs(last_week_actual - last_week_predictions)
# Define a threshold for anomaly detection (adjust as needed)
threshold = 0.1  # Example threshold: 10% difference
# Identify anomalies
anomalies = np.where(errors > threshold)[0]
# Plot the results with anomalies highlighted
plt.figure(figsize=(12, 6))
plt.plot(last_week_actual, label='Actual')
plt.plot(last_week_predictions, label='Predicted')
plt.scatter(anomalies, last_week_actual[anomalies], color='red', label='Anomalies')  # Highlight anomalies
plt.xlabel('Time Steps (last week)')
plt.ylabel('Pollution')
plt.title('Actual vs Predicted Pollution (Last Week) with Anomalies')
plt.legend()
# plt.show()


# 绘制实际值与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.6)
plt.plot([min(y_test_rescaled), max(y_test_rescaled)],
         [min(y_test_rescaled), max(y_test_rescaled)], 'r--', label='Ideal Fit')
plt.xlabel('Actual Pollution')
plt.ylabel('Predicted Pollution')
plt.title('Scatter Plot: Actual vs Predicted Pollution')
plt.legend()
# plt.show()

# 绘制残差图（残差 = 实际值 - 预测值）
residuals = y_test_rescaled - y_pred_rescaled
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_rescaled, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Pollution')
plt.ylabel('Residuals')
plt.title('Residual Plot')
# plt.show()

# 绘制残差分布直方图
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()