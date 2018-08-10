# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 读取数据集

data = pd.read_csv('/Users/alanp/Downloads/B0006.csv', header=0)
data = data.sort_values(by="count")
data_origin = data
train = data.drop(["id", "count", "state", "last_time", "vertical"], axis=1)
train_Y = train["soh"].values[:80]
train_X = train.drop(["soh"], axis=1).apply(
    lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).values[:80]
test_Y = train["soh"].values[80:]
test_X = train.drop(["soh"], axis=1).apply(
    lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).values[80:]
print(train_X.shape, train_Y.shape)
# REF为高斯核函数
kernel = RBF(0.5, (1e-4, 10))
# 创建高斯过程回归,并训练
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
reg.fit(train_X, train_Y)

output, err = reg.predict(test_X, return_std=True)
# 95%置信区间
up, down = output * (1 + 1.96 * err), output * (1 - 1.96 * err)
# 绘制结果
X = data_origin["count"][80:].values
plt.scatter(X, test_Y, label='real')
plt.scatter(X, output, label='predict', marker='*')
# X1 = data_origin["count"][:80]
# X2 = data_origin["count"][80:]
# z1 = np.polyfit(X1, Y, 3)
# z2 = np.polyfit(X, up, 3)
# z3 = np.polyfit(X, down, 3)
#
# Xvals = np.arange(-1.6, 0.3, 0.1)
# Yvals1 = np.polyval(z1, Xvals)
# Yvals2 = np.polyval(z2, Xvals)
# Yvals3 = np.polyval(z3, Xvals)
#
# plt.scatter(X, Y, label='data')
# plt.scatter(test, output, marker='*', label='new_data')
# plt.plot(Xvals, Yvals1, c='black', linestyle="--", label='fitting curve')
# plt.fill_between(Xvals, Yvals2, Yvals3, color='red', alpha=0.25)
plt.legend()
plt.show()
