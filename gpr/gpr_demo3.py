import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 使用高斯回归预测某曲线的未知值
X = np.array([-1.5, -1.0, -0.75, -0.4, -0.25, 0.0])
Y = np.array([-1.7, -1.2, -0.4, 0.1, 0.4, 0.6])
test = np.array([0.2])

kernel = RBF(0.5, (1e-4, 10))
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
reg.fit(X.reshape(-1, 1), Y.reshape(-1, 1))

output, err = reg.predict(test.reshape(-1, 1), return_std=True)

up, down = Y + abs(Y * 1.96 * err), Y - abs(Y * 1.96 * err)
z1 = np.polyfit(X, Y, 3)
z2 = np.polyfit(X, up, 3)
z3 = np.polyfit(X, down, 3)

Xvals = np.arange(-1.6, 0.3, 0.1)
Yvals1 = np.polyval(z1, Xvals)
Yvals2 = np.polyval(z2, Xvals)
Yvals3 = np.polyval(z3, Xvals)

plt.scatter(X, Y)
plt.scatter(test, output, marker='*')
plt.plot(Xvals, Yvals1, c='black', linestyle="--")
plt.fill_between(Xvals, Yvals2, Yvals3, color='red', alpha=0.25)
plt.show()
