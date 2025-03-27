import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import math
from math import sqrt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
import keras.backend as K

# 读取数据
df = pd.read_csv('./dataset/LSTM-Multivariate_pollution.csv')
# 将 'wnd_dir' 中的字符串映射为数值
wnd_dir_mapping = {
    'NE': 0,
    'NW': 1,
    'SE': 2,
    'cv': 3  # 如有其他方向可继续添加
}
df['wnd_dir'] = df['wnd_dir'].map(wnd_dir_mapping)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.head()

# 数据归一化
scaler = MinMaxScaler()
columns = ['pollution', 'dew', 'temp', 'press', "wnd_dir", 'wnd_spd', 'snow', 'rain']
df[columns] = scaler.fit_transform(df[columns])

# 准备训练数据（这里 n_past 和 n_future 可根据需求调整，确保CNN层输入长度足够）
df_train_scaled = np.array(df)
X = []
y = []
n_future = 1
n_past = 5  # 建议 n_past 设置大于1，以便 CNN 层和池化层能正常工作
for i in range(n_past, len(df_train_scaled) - n_future + 1):
    X.append(df_train_scaled[i - n_past:i, 0:df_train_scaled.shape[1]])
    y.append(df_train_scaled[i + n_future - 1:i + n_future, 0])
X, y = np.array(X), np.array(y)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print('X_train shape : {}   y_train shape : {} \nX_test shape : {}      y_test shape : {} '.format(
    X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# ---------------------- 构建 CNN + EA-LSTM 模型 ----------------------
model = Sequential()
# CNN 部分：提取局部时序特征
model.add(Conv1D(filters=64,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                 input_shape=(X_train.shape[1], X_train.shape[2])))
# 池化层：降低时间步维度（注意：如果 n_past 太小，pool_size 可调整为1）
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# LSTM 部分：处理序列数据（设置 return_sequences=True 以便后续 Attention 使用全部时间步的输出）
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))

# 全连接层
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 编译模型
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=[RootMeanSquaredError()])

model.summary()

# 定义回调函数，防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('CNN_LSTM_best_model.h5', monitor='val_loss', save_best_only=True)

# 训练模型
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, checkpoint])

# 加载最佳模型
best_model = load_model('CNN_LSTM_best_model.h5')

# 绘制训练和验证损失曲线
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# 预测与评估
y_pred = best_model.predict(X_test).flatten()
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# 将预测结果反归一化（这里只反归一化第一列数据）
y_pred_reshaped = y_pred.reshape(-1, 1)
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred_reshaped,
                                np.zeros((len(y_pred_reshaped), len(columns) - 1))], axis=1))[:, 0]
y_test_reshaped = y_test.reshape(-1, 1)
y_test_rescaled = scaler.inverse_transform(np.concatenate([y_test_reshaped,
                                np.zeros((len(y_test_reshaped), len(columns) - 1))], axis=1))[:, 0]
rmse_rescaled = math.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print('RMSE (Rescaled):', rmse_rescaled)

r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print('R-squared:', r2)

# 绘制最近一个月的实际值与预测值对比图
last_month_data = df.iloc[-744:]
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

# 绘制最近一周的实际值与预测值对比图
last_week_data = df.iloc[-168:]
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

# 绘制异常检测图：计算实际值与预测值的绝对误差，并将超过阈值的点标记为异常
errors = np.abs(last_week_actual - last_week_predictions)
threshold = 0.1  # 异常检测阈值（可根据实际情况调整）
anomalies = np.where(errors > threshold)[0]
plt.figure(figsize=(12, 6))
plt.plot(last_week_actual, label='Actual')
plt.plot(last_week_predictions, label='Predicted')
plt.scatter(anomalies, last_week_actual[anomalies], color='red', label='Anomalies')
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