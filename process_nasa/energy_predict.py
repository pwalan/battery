import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import math
from dateutil import parser
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, DotProduct, Matern, RationalQuadratic, \
    ExpSineSquared
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


def process_data(data, vid=0):
    if vid != 0:
        data = data[data['vehicle_id'] == vid]
    # else:
    #     dummies = pd.get_dummies(data['vehicle_id'], prefix='vehicle_id')
    #     data = data.join(dummies)
    # data = data.dropna()
    data = data.dropna()
    if 'charge_energy' in data.columns:
        data = data[data['charge_energy'] > 1]
    data = data[data['charge_start_soc'] != 100]
    data = data[data['charge_end_soc'] != 0]
    data = data[data['charge_end_soc'] > data['charge_start_soc']]
    data = data[data['charge_end_U'] > 50]
    data['time'] = (
            pd.to_datetime(data['charge_end_time'], format='%Y%m%d%H%M%S') - pd.to_datetime(data['charge_start_time'],
                                                                                            format='%Y%m%d%H%M%S')).dt.seconds
    data['C'] = -data['time'] * data['charge_start_I']
    data['W'] = -data['charge_start_U'] * data['charge_start_I'] / 1000
    # data['R'] = data['charge_start_U'] / data['charge_start_I']
    data['add_soc'] = data['charge_end_soc'] - data['charge_start_soc']

    # 多项式特征
    # column_names = ['mileage', 'time', 'charge_start_soc', 'charge_start_U']
    # features = data[column_names]
    # data = data.drop(column_names, axis=1)
    # poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    # poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
    #                              columns=poly_transformer.get_feature_names(column_names))
    # data = data.join(poly_features)

    # print(data.describe())
    return data


# 不做归一化
def get_data(vid, test_size):
    data = pd.read_csv(
        '/Users/alanp/Competition/创客工厂/2018全国高校新能源汽车大数据创新创业大赛/数据/创新题/energy_predict_data/finals/energy_train1029.csv',
        header=0)
    data = process_data(data, vid)
    data_origin = data
    train_data = data.drop(
        ['charge_energy', 'vehicle_id', 'charge_end_time', 'charge_end_soc', 'charge_start_time'], axis=1)
    train_Y = data["charge_energy"][:-test_size]
    train_X = train_data[:-test_size]
    test_Y = data["charge_energy"][-test_size:]
    test_X = train_data[-test_size:]

    valid_data = pd.read_csv(
        '/Users/alanp/Competition/创客工厂/2018全国高校新能源汽车大数据创新创业大赛/数据/创新题/energy_predict_data/finals/energy_test1029.csv',
        header=0)
    valid_data = process_data(valid_data, vid)
    valid_data_origin = valid_data
    valid_X = valid_data.drop(
        ['vehicle_id', 'charge_end_time', 'charge_start_time', 'charge_end_soc'],
        axis=1)
    return data, valid_data_origin, train_X, train_Y, test_X, test_Y, valid_X


def get_data_all():
    data = pd.read_csv(
        '/Users/alanp/Competition/创客工厂/2018全国高校新能源汽车大数据创新创业大赛/数据/创新题/energy_predict_data/finals/energy_train1029.csv',
        header=0)
    data = process_data(data)
    # data.to_csv('train.csv', index=False)
    data_origin = data
    features = ['charge_energy', 'vehicle_id', 'charge_end_time', 'charge_start_time', 'charge_start_I', 'charge_end_I']
    # train_X = data.drop(features, axis=1)
    # train_Y = data_origin["charge_energy"]
    data_X = data.drop(features, axis=1)
    data_Y = data_origin["charge_energy"]
    train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.05, random_state=27)

    # data = pd.read_csv(
    #     '/Users/alanp/Competition/创客工厂/2018全国高校新能源汽车大数据创新创业大赛/数据/创新题/energy_predict_data/processed/predict_data_e_test.csv',
    #     header=0)
    # data = process_data(data)
    # # data.to_csv('test.csv', index=False)
    # data_origin = data
    # test_X = data.drop(features, axis=1)
    # test_Y = data_origin["charge_energy"]

    valid_data = pd.read_csv(
        '/Users/alanp/Competition/创客工厂/2018全国高校新能源汽车大数据创新创业大赛/数据/创新题/energy_predict_data/finals/energy_test1029.csv',
        header=0)
    valid_data = process_data(valid_data)
    valid_data_origin = valid_data
    valid_X = valid_data.drop(
        ['vehicle_id', 'charge_end_time', 'charge_start_time', 'charge_start_I', 'charge_end_I'],
        axis=1)
    return data, valid_data_origin, train_X, train_Y, test_X, test_Y, valid_X


# 进行归一化
def get_data2(vid, test_size):
    data = pd.read_csv(
        '/Users/alanp/Competition/创客工厂/2018全国高校新能源汽车大数据创新创业大赛/数据/创新题/energy_predict_data/predict_data_e_train.csv',
        header=0)
    data = process_data(vid, data)
    data_origin = data
    train_data = data.drop(
        ['charge_energy', 'vehicle_id', 'charge_end_time', 'charge_start_time', 'charge_end_soc', 'charge_start_I',
         'charge_end_I'], axis=1)
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    train_Y = data["charge_energy"][:-test_size]
    train_X = train_data[:-test_size]
    test_Y = data["charge_energy"][-test_size:]
    test_X = train_data[-test_size:]

    valid_data = pd.read_csv(
        '/Users/alanp/Competition/创客工厂/2018全国高校新能源汽车大数据创新创业大赛/数据/创新题/energy_predict_data/testA.csv',
        header=0)
    valid_data = process_data(vid, valid_data)
    valid_data_origin = valid_data
    valid_X = valid_data.drop(
        ['vehicle_id', 'charge_end_time', 'charge_start_time', 'charge_end_soc', 'charge_start_I', 'charge_end_I'],
        axis=1)
    valid_X = scaler.transform(valid_X)
    return data, valid_data_origin, train_X, train_Y, test_X, test_Y, valid_X


def evalerror(preds, real):
    return math.sqrt(np.sum(((preds - real) / real) ** 2))


def xgboost_train(vid, test_size):
    data, valid_data, train_X, train_Y, test_X, test_Y, valid_X = get_data(vid, test_size)
    # model = xgb.XGBRegressor(max_depth=20, learning_rate=0.05, n_estimators=4000, silent=True,
    #                          objective='reg:gamma',
    #                          eval_metric='rmse')
    # model.fit(train_X, train_Y)
    #
    # # fig, ax = plt.subplots(figsize=(15, 15))
    # # xgb.plot_importance(model, height=0.5, ax=ax, )
    #
    # output = model.predict(test_X)
    # print(str(vid) + ' ' + str(evalerror(output, test_Y)))
    # predict = model.predict(valid_X)
    # res = pd.DataFrame({'vehicle_id': valid_data['vehicle_id'], 'charge_energy': predict})
    dtrain = xgb.DMatrix(train_X, label=train_Y.values)
    dtest = xgb.DMatrix(test_X)
    params = {'booster': 'gbtree',
              'objective': 'reg:gamma',
              'max_depth': 10,
              'eta': 0.05,
              'alpha': 0.01,
              'silent': 1
              }
    # watchlist = [(dtrain, 'train')]
    watchlist = []
    num_rounds = 1000
    model = xgb.train(params, dtrain, num_rounds, watchlist, feval=customedscore)

    # fig, ax = plt.subplots(figsize=(15, 15))
    # xgb.plot_importance(model, height=0.5, ax=ax, )

    dvalid = xgb.DMatrix(valid_X)
    predict = model.predict(dvalid)
    output = model.predict(dtest)
    print(str(vid) + ' ' + str(evalerror(output, test_Y)))
    res = pd.DataFrame()
    res['vehicle_id'] = valid_data['vehicle_id']
    res['charge_energy'] = predict
    return res, test_Y, output


def customedscore(preds, dtrain):
    label = dtrain.get_label()
    return 'error', math.sqrt(np.sum(((preds - label) / label) ** 2))


def xgboost_train2(vid, test_size):
    data, valid_data, train_X, train_Y, test_X, test_Y, valid_X = get_data(vid, test_size)
    dtrain = xgb.DMatrix(train_X, label=train_Y.values)
    dtest = xgb.DMatrix(test_X, label=test_Y.values)
    params = {'booster': 'gbtree',
              'objective': 'reg:gamma',
              'max_depth': 20,
              'eta': 0.05,
              'alpha': 0.01,
              'silent': 1
              }
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    num_rounds = 400
    model = xgb.train(params, dtrain, num_rounds, watchlist, feval=customedscore)

    # fig, ax = plt.subplots(figsize=(15, 15))
    # xgb.plot_importance(model, height=0.5, ax=ax, )

    dvalid = xgb.DMatrix(valid_X)
    predict = model.predict(dvalid)
    res = pd.DataFrame({'vehicle_id': valid_data['vehicle_id'], 'charge_energy': predict})
    return res


def xgboost_train3():
    data, valid_data, train_X, train_Y, test_X, test_Y, valid_X = get_data_all()
    dtrain = xgb.DMatrix(train_X, label=train_Y.values)
    dtest = xgb.DMatrix(test_X, label=test_Y.values)
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'max_depth': 10,
              'eta': 0.05,
              'alpha': 0.01,
              'lambda': 0.01,
              'silent': 1
              }
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    num_rounds = 1000
    model = xgb.train(params, dtrain, num_rounds, watchlist, feval=customedscore)

    fig, ax = plt.subplots(figsize=(15, 15))
    xgb.plot_importance(model, height=0.5, ax=ax, )

    dvalid = xgb.DMatrix(valid_X)
    predict = model.predict(dvalid)
    res = pd.DataFrame()
    res['vehicle_id'] = valid_data['vehicle_id']
    res['charge_energy'] = predict
    return res


def lgb():
    data, valid_data, train_X, train_Y, test_X, test_Y, valid_X = get_data_all()
    lgb_train = lgb.Dataset(train_X, train_Y)
    lgb_test = lgb.Dataset(test_X, test_Y, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 50,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_test,
                    early_stopping_rounds=5)

    y_pred = gbm.predict(test_X, num_iteration=gbm.best_iteration)
    predict = gbm.predict(valid_X, num_iteration=gbm.best_iteration)
    res = pd.DataFrame()
    res['vehicle_id'] = valid_data['vehicle_id']
    res['charge_energy'] = predict
    return res


def xgboost_train_all(train_X, train_Y, test_X, test_Y, valid_X, valid_data):
    model = xgb.XGBRegressor(max_depth=30, learning_rate=0.05, n_estimators=3000, silent=True,
                             objective='reg:gamma',
                             eval_metric='rmse')
    model.fit(train_X, train_Y)

    # fig, ax = plt.subplots(figsize=(15, 15))
    # xgb.plot_importance(model, height=0.5, ax=ax, )

    output = model.predict(test_X)
    print('All2:' + str(evalerror(output, test_Y)))
    predict = model.predict(valid_X)
    res = pd.DataFrame({'vehicle_id': valid_data['vehicle_id'], 'charge_energy': predict})
    return res


def elasticnet_train(vid, test_size):
    data, valid_data, train_X, train_Y, test_X, test_Y, valid_X = get_data(vid, test_size)
    model = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
                                      l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(train_X, train_Y)
    output = model.predict(test_X)
    print(str(vid) + ' ' + str(evalerror(output, test_Y)))

    predict = model.predict(valid_X)
    res = pd.DataFrame({'vehicle_id': valid_data['vehicle_id'], 'charge_energy': predict})
    return res, test_Y, output


def gpr_train(vid, test_size):
    data, valid_data, train_X, train_Y, test_X, test_Y, valid_X = get_data2(vid, test_size)
    # Matern RationalQuadratic ExpSineSquared DotProduct
    kernel = RBF()

    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    model.fit(train_X, train_Y)

    output, err = model.predict(test_X, return_std=True)
    print(str(vid) + ' ' + str(evalerror(output, test_Y)))
    predict = model.predict(valid_X)
    res = pd.DataFrame({'vehicle_id': valid_data['vehicle_id'], 'charge_energy': predict})
    return res, test_Y, output


def train_peocess():
    vids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    test_size = 5
    submission = pd.DataFrame()
    test_Ys = np.array([])
    outputs = np.array([])
    for vid in vids:
        res, test_Y, output = xgboost_train(vid, test_size)
        test_Ys = np.append(test_Ys, test_Y)
        outputs = np.append(outputs, output)
        submission = pd.concat([submission, res])
    # submission.to_csv('energy-submit.csv', index=False)
    print('All:' + str(evalerror(outputs, test_Ys)) + '\n')
    # plt.show()


def train_peocess2():
    vids = [1, 2, 3, 4, 5]
    test_size = 5
    submission = pd.DataFrame()
    for vid in vids:
        res = xgboost_train2(vid, test_size)
        submission = pd.concat([submission, res])
    # submission.to_csv('submit_A.csv', index=False)
    plt.show()


def train_peocess3():
    submission = xgboost_train3()
    # submission = lgb()
    submission.to_csv('energy-submit.csv', index=False)
    # plt.show()


if __name__ == "__main__":
    train_peocess3()
