import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, Sequential, callbacks
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler


df = pd.read_csv('./dataset/LSTM-Multivariate_pollution.csv')
df = df.set_index("date")
df.index = pd.to_datetime(df.index)

train = df.loc[df.index < "01-01-2014"]
test = df.loc[df.index >= "01-01-2014"]


def create_features(df):

    df = df.copy()

    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["quarter"] = df.index.quarter
    df["weekofyear"] = df.index.isocalendar().week

    return df
df = create_features(df)

fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(data=df, x="weekofyear", y="temp")

fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(data=df, x="month", y="pollution")


train = create_features(train)
test = create_features(test)

features = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow',
       'rain', 'hour', 'day', 'month', 'dayofweek', 'dayofyear', 'quarter',
       'weekofyear']
target = 'pollution'

df["wnd_dir"] = df["wnd_dir"].astype("category")

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

X_train["wnd_dir"] = X_train["wnd_dir"].astype("category")
X_test["wnd_dir"] = X_test["wnd_dir"].astype("category")
X_train.info()

model = xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=50, learning_rate=0.001, enable_categorical=True)

model.fit(X_train, y_train,
      eval_set=[(X_train, y_train), (X_test, y_test)],
      verbose=100)
# Load the best model
model.save_model('XGBoost_best_model.h5')

#importance
fi = pd.DataFrame(data=model.feature_importances_, index=model.feature_names_in_, columns=["importance"]).sort_values("importance")
fi.plot(kind="barh", title="Feature Importance")

y_pred = model.predict(X_test)


# Evaluate the model
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Replace object values in 'wnd_dir' with numerical values
wnd_dir_mapping = {
    'NE': 0,
    'NW': 1,
    'SE': 2,
    'cv': 3  # Add other directions as needed
}
df['wnd_dir'] = df['wnd_dir'].replace(wnd_dir_mapping)
scaler = MinMaxScaler()
columns_to_scale = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# Reshape y_pred to have the same number of features the scaler was trained on
y_pred_reshaped = y_pred.reshape(-1, 1)
# Concatenate with zeros to match the original number of features before scaling
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred_reshaped, np.zeros((len(y_pred_reshaped), len(columns_to_scale) - 1))], axis=1))[:, 0]
# Reshape y_test to have the same number of features the scaler was trained on
y_test = y_test.to_numpy()
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
