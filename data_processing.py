import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from pandas.plotting import autocorrelation_plot
from windrose import WindroseAxes
# from distributed.utils import palette

df_train = pd.read_csv('./dataset/LSTM-Multivariate_pollution.csv')
# print(df_train)
df_test = pd.read_csv('./dataset/pollution_test_data1.csv')
# print(df_test)
df_train_scaled = df_train.copy()
df_test_scaled = df_test.copy()

# Define the mapping dictionary
# mapping = {'NE': 0, 'SE':1 , 'NW': 2, 'cv': 3}
# mapping = {'NE': 45, 'SE':135 , 'NW': 315, 'cv': np.NAN}
mapping = {'NE': 45, 'SE':135 , 'NW': 315, 'cv': 225}

# Replace the string values with numerical values
df_train_scaled['wnd_dir'] = df_train_scaled['wnd_dir'].map(mapping)
df_test_scaled['wnd_dir'] = df_test_scaled['wnd_dir'].map(mapping)

df_train_scaled['date'] = pd.to_datetime(df_train_scaled['date'])
# Resetting the index
df_train_scaled.set_index('date', inplace=True)
print(df_train_scaled.head())
# correlation analysis
num_features = df_train_scaled.select_dtypes(include=['number']).columns[-8:]
correlation_matrix = df_train_scaled[num_features].corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap of Last 8 Features")
plt.tight_layout()
plt.show()

values = df_train_scaled.values
# print(values[3,0])
# specify columns to plot
groups = [0,1, 2, 3,5, 6,7]
i = 1

# plot each column
plt.figure(figsize=(20,14))
plt.subplots_adjust(hspace=0.4)
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(df_train_scaled.index,values[:, group], color=cm.viridis(group/len(groups)))
    plt.xlabel('Time')
    plt.title(df_train.columns[group+1], y=0.75, loc='right', fontsize = 15)
    i += 1
plt.tight_layout()
plt.show()

print(type(df_train_scaled))
print(df_train_scaled.columns)


# sns.set(style="darkgrid")

# fig, axs = plt.subplots(3,2, figsize=(24,14),hspace=)

# sns.histplot(data=df_train_scaled, x="pollution", kde=True, color="skyblue", ax=axs[0, 0])
# sns.histplot(data=df_train_scaled, x="dew", kde=True, color="olive", ax=axs[0, 1])
# sns.histplot(data=df_train_scaled, x="temp", kde=True, color="gold", ax=axs[1, 0])
# sns.histplot(data=df_train_scaled, x="press", kde=True, color="teal", ax=axs[1, 1])
# sns.histplot(data=df_train_scaled, x="snow", kde=True, color="steelblue", ax=axs[2, 0])
# sns.histplot(data=df_train_scaled, x="rain", kde=True, color="goldenrod", ax=axs[2, 1])
# plt.tight_layout()
# plt.show()


labels = ["N", "NE", "E", "SE", "S", "CV", "W", "NW"]
ax = WindroseAxes.from_ax()
ax.bar(df_train_scaled["wnd_dir"], df_train_scaled["wnd_spd"], normed=True, opening=0.8, edgecolor="white")
ax.set_legend()
ax.set_xticklabels(labels)
legend = ax.legend(loc="upper right", bbox_to_anchor=(1, 1.1))  # 右上角偏移
plt.tight_layout()
# plt.title("Wind Speed and Direction Distribution")
plt.show()

sns.pairplot(df_train_scaled[['pollution', 'temp', 'dew', 'press', 'wnd_spd']])
plt.show()



result = seasonal_decompose(df_train_scaled['pollution'], model='additive', period=365)
result.plot()
plt.show()

color=["skyblue","olive","gold","teal",'lightblue']
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_train_scaled[['pollution', 'temp', 'dew', 'press', 'wnd_spd']],palette=color)
# plt.title("Box Plot of Key Features")
plt.tight_layout()
plt.show()



df_train_scaled["pollution_MA"] = df_train_scaled["pollution"].rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(df_train_scaled.index, df_train_scaled["pollution"], label="Original Data", alpha=0.5)
plt.plot(df_train_scaled.index, df_train_scaled["pollution_MA"], label="30-Day Moving Average", color="red")
plt.xlabel("Date")
plt.ylabel("Pollution Level")
plt.title("Pollution Trends with Moving Average")
plt.legend()
plt.show()




plt.figure(figsize=(12, 3))
autocorrelation_plot(df_train_scaled["pollution"])
plt.tight_layout()
# plt.title("Autocorrelation of Pollution Data")
plt.show()




pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_train_scaled.select_dtypes(include=['number']))
df_train_scaled['pca-1'] = pca_result[:, 0]
df_train_scaled['pca-2'] = pca_result[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x="pca-1", y="pca-2", data=df_train_scaled)
# plt.title("PCA Projection of Features")
plt.tight_layout()
plt.show()


print("Explained variance ratio:", pca.explained_variance_ratio_)#[0.75857606 0.20380579]





# 创建图像
fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True, gridspec_kw={'hspace': 0.1})

# 设置全局风格
sns.set_style("whitegrid")
pollution_data=df_train_scaled["pollution"]
pollution_moving_avg = pollution_data.rolling(window=30).mean()
axs[0].plot(pollution_data, alpha=0.6, linewidth=1, label="Original Data")
axs[0].plot(pollution_moving_avg, color="brown", linewidth=1.5, label="30-Day Moving Average")
# axs[0].set_title("Pollution Trends with Moving Average", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Pollution Level with Moving Average")
axs[0].legend()


decomposition = seasonal_decompose(pollution_data, model='additive', period=365)
# 2. 趋势分解
axs[1].plot(decomposition.trend,  linewidth=1.5)
# axs[1].set_title("Trend", fontsize=14, fontweight='bold')
axs[1].set_ylabel("Trend Value")

# 3. 季节性成分
axs[2].plot(decomposition.seasonal,  linewidth=1)
# axs[2].set_title("Seasonal Component", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Seasonality")

# 4. 残差部分
axs[3].scatter(pollution_data.index, decomposition.resid,  s=5, alpha=0.6)
# axs[3].set_title("Residual Component", fontsize=14, fontweight='bold')
axs[3].set_ylabel("Residual")

# # 5. 自相关分析
# autocorrelation_plot(pollution_data, ax=axs[4])
# axs[4].set_title("Autocorrelation of Pollution Data", fontsize=14, fontweig with Moving Averageht='bold')
# axs[4].set_ylabel("Autocorrelation")

# 美化整体布局
plt.xlabel("Time")
sns.despine()
plt.tight_layout()

# 显示图像
plt.show()