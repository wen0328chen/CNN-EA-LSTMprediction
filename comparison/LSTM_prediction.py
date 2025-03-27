import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import seaborn as sns
import math
from math import sqrt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.metrics import mean_squared_error as mse, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


df = pd.read_csv('./dataset/LSTM-Multivariate_pollution.csv')
# print(df.head())



# Replace object values in 'wnd_dir' with numerical values
wnd_dir_mapping = {
    'NE': 0,
    'NW': 1,
    'SE': 2,
    'cv': 3  # Add other directions as needed
}
df['wnd_dir'] = df['wnd_dir'].map(wnd_dir_mapping)

df['date'] = pd.to_datetime(df['date'])
# Resetting the index
df.set_index('date', inplace=True)
df.head()


scaler = MinMaxScaler()
# Define the columns to scale
columns = (['pollution', 'dew', 'temp', 'press', "wnd_dir", 'wnd_spd',
            'snow', 'rain'])
# Scale the selected columns to the range 0-1
df[columns] = scaler.fit_transform(df[columns])


df_train_scaled = np.array(df)
X = []
y = []
n_future = 1
n_past = 5
#  Train Sets
for i in range(n_past, len(df_train_scaled) - n_future+1):
    X.append(df_train_scaled[i - n_past:i, 0:df_train_scaled.shape[1]])
    y.append(df_train_scaled[i + n_future - 1:i + n_future, 0])
X, y = np.array(X), np.array(y)
#  Test Sets
# split into train and test sets:
train_size = int(len(X) * 0.8)  # Example: 80% for training
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

print('X_train shape : {}   y_train shape : {} \n'
      'X_test shape : {}      y_test shape : {} '.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))


# design network
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
# Define callbacks for avoiding overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('LSTM_best_model.h5', monitor='val_loss', save_best_only=True)
model.summary()


# fit network
# with tf.device('/GPU:0'):
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data= (X_test,y_test), callbacks=[early_stopping, checkpoint])


# Load the best model
best_model = load_model('LSTM_best_model.h5')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


y_pred = best_model.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Train Predictions': y_pred,
                                  'Actual':y_test.flatten()})
# test_results.head()
# # Make predictions
# y_pred = model.predict(X_test)
# Evaluate the model
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Reshape y_pred to have the same number of features the scaler was trained on
y_pred_reshaped = y_pred.reshape(-1, 1)
# Concatenate with zeros to match the original number of features before scaling
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred_reshaped, np.zeros((len(y_pred_reshaped), len(columns) - 1))], axis=1))[:, 0]
# Reshape y_test to have the same number of features the scaler was trained on
y_test_reshaped = y_test.reshape(-1, 1)
# Concatenate with zeros to match the original number of features before scaling
y_test_rescaled = scaler.inverse_transform(np.concatenate([y_test_reshaped, np.zeros((len(y_test_reshaped), len(columns) - 1))], axis=1))[:, 0]
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