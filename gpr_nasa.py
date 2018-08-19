# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# 读取数据集
bid = 'B0006'
data = pd.read_csv('/Users/alanp/Downloads/battery/' + bid + '.csv', header=0)
data = data.sort_values(by="count")
data_origin = data
train = data.drop(["id", "count", "state", "last_time", "vertical"], axis=1)
# train = data.drop(["id", "count", "state", "last_time"], axis=1)
sp = MinMaxScaler()
# train_data = train.drop(["soh"], axis=1)
train_data = sp.fit_transform(train.drop(["soh"], axis=1))
# train_data = train.drop(["soh"], axis=1).apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).values
train_size = 80
train_Y = train["soh"][:train_size]
train_X = train_data[:train_size]
test_Y = train["soh"][train_size:]
test_X = train_data[train_size:]
print(train_X.shape, train_Y.shape)
# REF为高斯核函数
# kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10)) + C()
kernel = RBF(0.5, (1e-4, 5)) + DotProduct(0.1, (0.001, 0.1))
# kernel = RBF()
# 创建高斯过程回归,并训练
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
reg.fit(train_X, train_Y)
# 预测
output, err = reg.predict(test_X, return_std=True)
# RMSE
rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
print(rmse)
# 95%置信区间
total = np.array(list(train_Y) + list(output))
err = np.append(np.zeros(train_size) + 0.05, err)
up, down = total * (1 + 1.96 * err), total * (1 - 1.96 * err)
# 绘制结果
# X = data_origin["count"][80:].values
X = np.arange(data.shape[0])
plt.scatter(X, train["soh"].values, label='real')
plt.scatter(X, total, label='predict', marker='*')
plt.fill_between(X, up, down, color='red', alpha=0.25)
plt.title("RMSE on " + bid + " is: " + str(rmse))
plt.legend()
plt.show()
