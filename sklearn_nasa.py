# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, Matern, RationalQuadratic, \
    ExpSineSquared
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import linear_model
import xgboost as xgb
from xgboost import plot_importance


def get_data(bid, train_size):
    # 读取编号为bid的数据集，通过train_size划分训练集和测试集
    data = pd.read_csv('/Users/alanp/Downloads/battery/' + bid + '.csv', header=0)
    data = data.sort_values(by="count")
    data_origin = data
    sp = MinMaxScaler()
    # train_data = sp.fit_transform(data.drop(["id", "count", "state", "last_time", "vertical"], axis=1))
    train_data = sp.fit_transform(data[["cc_duration", "cv_duration", "slope", "vertical"]])
    train_Y = data["soh"][:train_size]
    train_X = train_data[:train_size]
    test_Y = data["soh"][train_size:]
    test_X = train_data[train_size:]
    return data, train_X, train_Y, test_X, test_Y


def get_data2(train_bids, test_bid):
    test_data = pd.read_csv('/Users/alanp/Downloads/battery/' + test_bid + '.csv', header=0)
    test_Y = test_data["soh"]
    test_X = MinMaxScaler().fit_transform(test_data[["cc_duration", "cv_duration", "slope", "vertical"]])
    train_Y = test_Y
    train_X = test_X
    isFirst = True
    data = test_data
    for bid in train_bids:
        if isFirst:
            data = pd.read_csv('/Users/alanp/Downloads/battery/' + bid + '.csv', header=0)
            isFirst = False
        else:
            data.append(pd.read_csv('/Users/alanp/Downloads/battery/' + bid + '.csv', header=0))
    train_Y = data["soh"]
    train_X = MinMaxScaler().fit_transform(data[["cc_duration", "cv_duration", "slope", "vertical"]])
    return test_data, train_X, train_Y, test_X, test_Y


def gpr_train(bid, train_size):
    data, train_X, train_Y, test_X, test_Y = get_data(bid, train_size)
    # 核函数
    # kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10)) + C()
    # kernel = RBF(0.5, (1e-4, 5)) + DotProduct(0.1, (0.001, 0.1))
    kernel = RBF() + Matern() + DotProduct()
    # 创建高斯过程回归,并训练
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    reg.fit(train_X, train_Y)
    # 预测
    output, err = reg.predict(test_X, return_std=True)
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    print(bid + ": " + str(rmse))
    # 95%置信区间
    total = np.array(list(train_Y) + list(output))
    err = np.append(np.zeros(train_size) + 0.05, err)
    up, down = total * (1 + 1.96 * err), total * (1 - 1.96 * err)
    X = np.arange(data.shape[0])
    plt.fill_between(X, up, down, color='red', alpha=0.25)
    # plt.scatter(X, data["soh"].values, label='real')
    # plt.scatter(X, total, label='predict', marker='*')
    # plt.title(bid + ":" + str(rmse))
    # plt.legend()
    return X, data, total, rmse


def gpr_train2(train_bids, test_bid):
    test_data, train_X, train_Y, test_X, test_Y = get_data2(train_bids, test_bid)
    # 核函数
    kernel = RBF() + Matern() + DotProduct()
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    reg.fit(train_X, train_Y)
    output, err = reg.predict(test_X, return_std=True)
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    print(test_bid + ": " + str(rmse))
    X = np.arange(test_data.shape[0])
    up, down = output * (1 + 1.96 * err), output * (1 - 1.96 * err)
    return X, test_Y, output, rmse, up, down


def xgboost_train(bid, train_size):
    data, train_X, train_Y, test_X, test_Y = get_data(bid, train_size)
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=500, silent=False,
                             objective='reg:gamma',
                             eval_metric='rmse')
    model.fit(train_X, train_Y)
    output = model.predict(test_X)
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    X = np.arange(data.shape[0])
    total = np.array(list(train_Y) + list(output))
    return X, data, total, rmse


def elasticnet_train(bid, train_size):
    data, train_X, train_Y, test_X, test_Y = get_data(bid, train_size)
    elasticnet = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
                                           l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(train_X, train_Y)
    # 利用模型预测，x_test为测试集特征变量
    output = elasticnet.predict(test_X)
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    X = np.arange(data.shape[0])
    total = np.array(list(train_Y) + list(output))
    return X, data, total, rmse


def gbdt_train(bid, train_size):
    data, train_X, train_Y, test_X, test_Y = get_data(bid, train_size)
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0, loss='ls').fit(
        train_X, train_Y)
    output = est.predict(test_X)
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    X = np.arange(data.shape[0])
    total = np.array(list(train_Y) + list(output))
    return X, data, total, rmse


def train_peocess():
    bids = ['B0005', 'B0006', 'B0007', 'B0033']
    train_size = 80
    fig = plt.figure()
    i = 1
    for bid in bids:
        plt.subplot(2, 2, i)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        X, data, total, rmse = elasticnet_train(bid, train_size)
        plt.scatter(X, data["soh"].values, label='real')
        plt.scatter(X, total, label='predict', marker='*')
        plt.title(bid + ":" + str(rmse))
        plt.legend()
        i += 1
    plt.show()


def train_peocess2():
    bids = ['B0005', 'B0006', 'B0007', 'B0033']
    fig = plt.figure()
    i = 1
    for bid in bids:
        tmp_bids = bids.copy()
        tmp_bids.remove(bid)
        plt.subplot(2, 2, i)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        X, test_Y, output, rmse, up, down = gpr_train2(tmp_bids, bid)
        plt.fill_between(X, up, down, color='red', alpha=0.25)
        plt.scatter(X, test_Y.values, label='real')
        plt.scatter(X, output, label='predict', marker='*')
        plt.title(bid + ":" + str(rmse))
        plt.legend()
        i += 1
    plt.show()


if __name__ == "__main__":
    train_peocess()
