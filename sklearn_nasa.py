# coding: utf-8
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import xgboost as xgb
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, DotProduct, Matern, RationalQuadratic, \
    ExpSineSquared
from sklearn.preprocessing import MinMaxScaler


def get_data(bid, train_size):
    # 读取编号为bid的数据集，通过train_size划分训练集和测试集
    data = pd.read_csv('/Users/alanp/Downloads/battery/' + bid + '.csv', header=0)
    data = data.sort_values(by="count")
    data_origin = data
    sp = MinMaxScaler()
    # train_data = sp.fit_transform(data.drop(["id", "count", "state", "last_time", "soh"], axis=1))
    # train_data = sp.fit_transform(data[["cc_duration", "cv_duration", "vertical", "slope"]])
    train_data = sp.fit_transform(data[["cc_duration", "cv_duration", "vertical", "slope", "cc_slope", "f"]])
    # train_data = sp.fit_transform(data[["cc_duration", "cv_duration", "slope", "vertical"]])
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
    # Matern RationalQuadratic ExpSineSquared
    kernel = RBF() + Matern() + RationalQuadratic() + DotProduct()
    # kernel = RBF()
    # kernel = RBF()  + 0.5*RationalQuadratic() + 0.5*DotProduct()
    # 创建高斯过程回归,并训练
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    time_start = time.time()
    reg.fit(train_X, train_Y)
    print("train: " + str(time.time() - time_start) + "s")
    # 预测
    time_start = time.time()
    output, std = reg.predict(test_X, return_cov=True)
    print("predict: " + str(time.time() - time_start) + "s")
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    print(bid + ": " + str(rmse))
    # 95%置信区间
    total = np.array(list(train_Y) + list(output))
    std = np.append(np.zeros(train_size) + 0.005, std[0])
    up, down = total + 1.96 * std, total - 1.96 * std
    # 计算置信区间
    avg_train = np.sum(train_Y) / len(train_Y)
    cov_train = np.std(train_Y)
    up_train = avg_train + 1.96 * cov_train
    down_train = avg_train - 1.96 * cov_train
    print("train 0.95 interval (" + str(down_train) + "," + str(up_train) + ")")
    avg_output = np.sum(output) / len(output)
    cov_output = np.std(output)
    up_output = avg_output + 1.96 * cov_output
    down_output = avg_output - 1.96 * cov_output
    print("predict 0.95 interval (" + str(down_output) + "," + str(up_output) + ")\n")
    # up = []
    # down = []
    # for i in range(len(train_Y)):
    #     up.append(train_Y[i] + 1.96 * cov_train)
    #     down.append(train_Y[i] - 1.96 * cov_train)
    # for i in range(len(output)):
    #     up.append(output[i] + 1.96 * cov_output)
    #     down.append(output[i] - 1.96 * cov_output)
    # 绘图
    # up, down = train_Y * (1 + err), train_Y * (1 - err)
    # up = list(up)
    # down = list(down)
    # cov = list(cov[0])
    # for tmp_cov in cov:
    #     up.append(mean_output + 1.96 * tmp_cov)
    #     down.append(mean_output - 1.96 * tmp_cov)
    X = np.arange(data.shape[0])
    # label='95% confidence interval'
    plt.fill_between(X, up, down, color='red', alpha=0.25)
    # plt.scatter(X, data["soh"].values, label='real')
    # plt.scatter(X, total, label='predict', marker='*')
    # plt.title(bid + ":" + str(rmse))
    # plt.legend()
    return X, data, total, rmse


def gpr_gridsearch():
    data, train_X, train_Y, test_X, test_Y = get_data('B0005', 80)
    w1 = w2 = w3 = w4 = w5 = 0.0
    min_rmse = 100000
    best_w = ""
    step = 1.0
    time_start = time.time()
    for w1 in np.arange(step, 1.0 + step, step):
        print("w1: " + str(w1))
        for w2 in np.arange(0, 1.0 + step, step):
            for w3 in np.arange(0, 1.0 + step, step):
                for w4 in np.arange(0, 1.0 + step, step):
                    for w5 in np.arange(0, 1.0 + step, step):
                        kernel = C(constant_value=w1) * RBF() + C(constant_value=w2) * Matern() + C(
                            constant_value=w3) * ExpSineSquared() + C(constant_value=w4) * RationalQuadratic() + C(
                            constant_value=w5) * DotProduct()
                        reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
                        reg.fit(train_X, train_Y)
                        output, err = reg.predict(test_X, return_std=True)
                        rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
                        print(str(w1) + ", " + str(w2) + ", " + str(w3) + ", " + str(w4) + ", " + str(w5) + ": " + str(
                            rmse))
                        if rmse < min_rmse:
                            min_rmse = rmse
                            best_w = str(w1) + ", " + str(w2) + ", " + str(w3) + ", " + str(w4) + ", " + str(w5)
    print("gridsearch use: " + str(time.time() - time_start) + "s")
    print(min_rmse)
    print(best_w)


def gpr_train2(train_bids, test_bid):
    test_data, train_X, train_Y, test_X, test_Y = get_data2(train_bids, test_bid)
    kernel = RBF() + Matern() + RationalQuadratic() + DotProduct()
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    reg.fit(train_X, train_Y)
    output, err = reg.predict(test_X, return_std=True)
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    print(test_bid + ": " + str(rmse))
    X = np.arange(test_data.shape[0])
    up, down = output * (1 + 1.96 * err), output * (1 - 1.96 * err)
    return X, test_Y, output, rmse, up, down


def gpr_train3(bid, train_size):
    data, train_X, train_Y, test_X, test_Y = get_data(bid, train_size)

    kernel = RBF() + Matern() + RationalQuadratic() + DotProduct()
    # kernel = RBF() + DotProduct()

    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

    time_start = time.time()
    reg.fit(train_X, train_Y)
    print("train: " + str(time.time() - time_start))
    time_start = time.time()

    output, err = reg.predict(test_X, return_std=True)
    print("predict: " + str(time.time() - time_start))

    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    print(bid + ": " + str(rmse))

    # 95%置信区间
    total = np.array(list(train_Y) + list(output))
    err = np.append(np.zeros(train_size) + 0.05, err)
    up, down = total * (1 + 0.95 * err), total * (1 - 0.95 * err)
    X = np.arange(data.shape[0])
    plt.fill_between(X, up, down, color='red', alpha=0.25)
    return X, data, total, rmse, test_Y - output


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
    time_start = time.time()
    elasticnet = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
                                           l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(train_X, train_Y)
    # elasticnet = linear_model.LassoCV().fit(train_X, train_Y)
    print("train: " + str(time.time() - time_start))
    # 利用模型预测，x_test为测试集特征变量
    time_start = time.time()
    output = elasticnet.predict(test_X)
    print("predict: " + str(time.time() - time_start))
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    print(bid + ": " + str(rmse))
    X = np.arange(data.shape[0])
    total = np.array(list(train_Y) + list(output))
    return X, data, total, rmse


def elasticnet_train2(train_bids, test_bid):
    test_data, train_X, train_Y, test_X, test_Y = get_data2(train_bids, test_bid)
    # elasticnet = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
    #                                        l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(train_X, train_Y)
    elasticnet = linear_model.LassoCV().fit(train_X, train_Y)
    # 利用模型预测，x_test为测试集特征变量
    output = elasticnet.predict(test_X)
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    X = np.arange(test_data.shape[0])
    total = np.array(list(train_Y) + list(output))
    up = []
    down = []
    return X, test_Y, output, rmse, up, down


def gbdt_train(bid, train_size):
    data, train_X, train_Y, test_X, test_Y = get_data(bid, train_size)
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0, loss='ls').fit(
        train_X, train_Y)
    output = est.predict(test_X)
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    X = np.arange(data.shape[0])
    total = np.array(list(train_Y) + list(output))
    return X, data, total, rmse


def randomforest_train(bid, train_size):
    data, train_X, train_Y, test_X, test_Y = get_data(bid, train_size)
    rfr = RandomForestRegressor(n_estimators=100).fit(train_X, train_Y)
    output = rfr.predict(test_X)
    rmse = np.sqrt(metrics.mean_squared_error(test_Y, output))
    X = np.arange(data.shape[0])
    total = np.array(list(train_Y) + list(output))
    return X, data, total, rmse


def train_peocess():
    # bids = ['B0005', 'B0006', 'B0007', 'B0033']
    bids = ['B0005', 'B0006', 'B0007', 'B0033', 'B0018', 'B0036']
    train_size = 80
    fig = plt.figure(figsize=(10, 10))
    i = 1
    for bid in bids:
        plt.subplot(3, 2, i)
        # plt.subplots_adjust(wspace=0.3, hspace=0.5)
        # plt.subplots_adjust(left=0, right=10, bottom=0, top=30)
        X, data, total, rmse = gpr_train(bid, train_size)
        plt.plot(X, data["soh"].values, label='Real')
        plt.plot(X, total, label='Estimate', c='red', linestyle='--')
        plt.axvline(train_size, c='green', linestyle=':')
        plt.xlabel('Cycle')
        plt.ylabel('SoH')
        # plt.text(0, 0.8, 'Train')
        # plt.text(train_size + 10, 0.8, 'Test')
        plt.title(bid + ":" + str(rmse))
        plt.legend(loc='lower left')
        i += 1
    fig.tight_layout()
    # plt.show()
    fig.savefig("train.png", dpi=200)


def train_peocess0():
    # bids = ['B0005', 'B0006', 'B0007', 'B0033']
    bids = ['B0005', 'B0006', 'B0007', 'B0033', 'B0018', 'B0036']
    train_size = 80
    fig, ax = plt.subplots(figsize=(12, 8), ncols=2, nrows=3)
    i = 0
    for bid in bids:
        # plt.subplot(3, 2, i)
        # plt.subplots_adjust(wspace=0.3, hspace=0.5)
        # plt.subplots_adjust(left=0.3)
        X, data, total, rmse = gpr_train(bid, train_size)
        ax[i].plot(X, data["soh"], label='Real')
        ax[i].plot(X, total, label='Estimate', c='red', linestyle='--')
        ax[i].axvline(train_size, c='green', linestyle=':')
        ax[i].xlabel('Cycle')
        ax[i].ylabel('SoH')
        # plt.text(0, 0.8, 'Train')
        # plt.text(train_size + 10, 0.8, 'Test')
        ax[i].title(bid + ":" + str(rmse))
        ax[i].legend(loc='lower left')
        i += 1
    fig.tight_layout()
    # plt.show()

    fig.savefig("train.png", dpi=200)


def test_peocess():
    bid = 'RW9'
    train_size = 40
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    X, data, total, rmse, err = gpr_train3(bid, train_size)
    plt.plot(X, data["soh"].values, label='Real')
    plt.plot(X, total, label='Estimate', c='red', linestyle='--')
    plt.axvline(train_size, c='green', linestyle=':')
    plt.xlabel('Cycle')
    plt.ylabel('SoH')
    # plt.text(0, 0.8, 'Train')
    # plt.text(train_size + 10, 0.8, 'Test')
    plt.title(bid + ":" + str(rmse))
    plt.legend(loc='lower left')

    plt.subplot(2, 1, 2)
    plt.scatter(X[train_size:], err)
    plt.yticks(np.arange(-0.2, 0.2, 0.05))
    plt.xlabel('Cycle')
    plt.ylabel('SoH error')
    # for a, b in zip(X[train_size:], err):
    #     plt.text(a, b+0.01, '%.3f' % b, ha='center', va='bottom', fontsize=5)
    plt.grid(True)
    plt.legend(loc='lower left')
    # fig.tight_layout()
    fig.set_dpi(200)
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
        # X, test_Y, output, rmse, up, down = gpr_train2(tmp_bids, bid)
        # plt.fill_between(X, up, down, color='red', alpha=0.25)
        X, test_Y, output, rmse, up, down = gpr_train2(tmp_bids, bid)
        plt.scatter(X, test_Y.values, label='real')
        plt.scatter(X, output, label='predict', marker='*')
        plt.title(bid + ":" + str(rmse))
        plt.legend()
        i += 1
    plt.show()


if __name__ == "__main__":
    # gpr_gridsearch()
    train_peocess()
    # test_peocess()
