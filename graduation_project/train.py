# coding:utf-8
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.datasets import imdb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib


def get_data():
    df = pd.read_csv("/Users/alanp/Downloads/bishedata/hot/train.csv", header=-1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # X_train = MinMaxScaler().fit_transform(X_train)
    # X_test = MinMaxScaler().fit_transform(X_test)
    return X_train, X_test, y_train, y_test


def get_test_data():
    df = pd.read_csv("/Users/alanp/Downloads/bishedata/hot/test.csv", header=-1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # X = MinMaxScaler().fit_transform(X)
    return X, y


def logisticRegression():
    X_train, X_test, y_train, y_test = get_data()
    lr = LogisticRegression()  # 实例化一个LR模型
    lr.fit(X_train, y_train)  # 训练模型
    y_prob = lr.predict_proba(X_test)[:, 1]  # 预测1类的概率
    y_pred = lr.predict(X_test)  # 模型对测试集的预测结果
    fpr_lr, tpr_lr, threshold_lr = metrics.roc_curve(y_test, y_prob)  # 获取真阳率、伪阳率、阈值
    auc_lr = metrics.auc(fpr_lr, tpr_lr)  # AUC得分
    score_lr = metrics.accuracy_score(y_test, y_pred)  # 模型准确率
    print([score_lr, auc_lr])
    # 实测
    X, y = get_test_data()
    y_prob = lr.predict_proba(X)[:, 1]
    y_pred = lr.predict(X)
    fpr_lr, tpr_lr, threshold_lr = metrics.roc_curve(y, y_prob)
    auc_lr = metrics.auc(fpr_lr, tpr_lr)
    score_lr = metrics.accuracy_score(y, y_pred)
    print([score_lr, auc_lr])


def svm():
    X_train, X_test, y_train, y_test = get_data()
    svc = SVC(kernel='rbf').fit(X_train, y_train)
    y_prob = svc.decision_function(X_test)
    y_pred = svc.predict(X_test)
    fpr_svc, tpr_svc, threshold_svc = metrics.roc_curve(y_test, y_prob)
    auc_svc = metrics.auc(fpr_svc, tpr_svc)
    score_svc = metrics.accuracy_score(y_test, y_pred)
    print([score_svc, auc_svc])
    # 实测
    X, y = get_test_data()
    y_prob = svc.decision_function(X)
    y_pred = svc.predict(X)
    fpr_svc, tpr_svc, threshold_svc = metrics.roc_curve(y, y_prob)
    auc_svc = metrics.auc(fpr_svc, tpr_svc)
    score_svc = metrics.accuracy_score(y, y_pred)
    print([score_svc, auc_svc])


def knn():
    X_train, X_test, y_train, y_test = get_data()
    knn = KNeighborsClassifier().fit(X_train, y_train)
    y_prob = knn.predict_proba(X_test)[:, 1]
    y_pred = knn.predict(X_test)
    fpr_knn, tpr_knn, threshold_knn = metrics.roc_curve(y_test, y_prob)
    auc_knn = metrics.auc(fpr_knn, tpr_knn)
    score_knn = metrics.accuracy_score(y_test, y_pred)
    print([score_knn, auc_knn])
    # 实测
    X, y = get_test_data()
    y_prob = knn.predict_proba(X)[:, 1]
    y_pred = knn.predict(X)
    fpr_svc, tpr_svc, threshold_svc = metrics.roc_curve(y, y_prob)
    auc_svc = metrics.auc(fpr_svc, tpr_svc)
    score_svc = metrics.accuracy_score(y, y_pred)
    print([score_svc, auc_svc])


def decision_tree():
    X_train, X_test, y_train, y_test = get_data()
    dtc = tree.DecisionTreeClassifier()  # 建立决策树模型
    dtc.fit(X_train, y_train)  # 训练模型
    y_prob = dtc.predict_proba(X_test)[:, 1]
    y_pred = dtc.predict(X_test)
    fpr_dtc, tpr_dtc, threshod_dtc = metrics.roc_curve(y_test, y_prob)
    score_dtc = metrics.accuracy_score(y_test, y_pred)
    auc_dtc = metrics.auc(fpr_dtc, tpr_dtc)
    print([score_dtc, auc_dtc])
    # 实测
    X, y = get_test_data()
    y_prob = dtc.predict_proba(X)[:, 1]
    y_pred = dtc.predict(X)
    fpr_dtc, tpr_dtc, threshod_dtc = metrics.roc_curve(y, y_prob)
    score_dtc = metrics.accuracy_score(y, y_pred)
    auc_dtc = metrics.auc(fpr_dtc, tpr_dtc)
    print([score_dtc, auc_dtc])


def random_forest():
    X_train, X_test, y_train, y_test = get_data()
    rfc = RandomForestClassifier()  # 建立随机森林分类器
    rfc.fit(X_train, y_train)  # 训练随机森林模型
    y_prob = rfc.predict_proba(X_test)[:, 1]
    y_pred = rfc.predict(X_test)
    fpr_rfc, tpr_rfc, threshold_rfc = metrics.roc_curve(y_test, y_prob)
    auc_rfc = metrics.auc(fpr_rfc, tpr_rfc)
    score_rfc = metrics.accuracy_score(y_test, y_pred)
    print([score_rfc, auc_rfc])
    # 实测
    X, y = get_test_data()
    y_prob = rfc.predict_proba(X)[:, 1]
    y_pred = rfc.predict(X)
    fpr_rfc, tpr_rfc, threshold_rfc = metrics.roc_curve(y, y_prob)
    auc_rfc = metrics.auc(fpr_rfc, tpr_rfc)
    score_rfc = metrics.accuracy_score(y, y_pred)
    print([score_rfc, auc_rfc])


def random_forest_tuning():
    X_train, X_test, y_train, y_test = get_data()
    # rfc = RandomForestClassifier()
    # tuned_parameters = [
    #     {'max_depth': [5, 10, 15, 20], 'min_samples_split': [1, 2, 3, 4], 'min_samples_leaf': [1, 2, 3, 4],
    #      'n_estimators': [50, 100, 150, 200]}]
    # clf = GridSearchCV(estimator=rfc, param_grid=tuned_parameters, cv=10, n_jobs=-1)
    # clf.fit(X_train, y_train)
    # print(clf.best_params_)
    rfc = RandomForestClassifier(max_depth=15, n_estimators=15, min_samples_leaf=1, min_samples_split=2)
    rfc.fit(X_train, y_train)
    print(rfc.feature_importances_)
    # joblib.dump(rfc, "/Users/alanp/Desktop/shortcircuit.m")
    y_prob = rfc.predict_proba(X_test)[:, 1]
    y_pred = rfc.predict(X_test)
    fpr_rfc, tpr_rfc, threshold_rfc = metrics.roc_curve(y_test, y_prob)
    auc_rfc = metrics.auc(fpr_rfc, tpr_rfc)
    score_rfc = metrics.accuracy_score(y_test, y_pred)
    print([score_rfc, auc_rfc])
    X, y = get_test_data()
    y_prob = rfc.predict_proba(X)[:, 1]
    y_pred = rfc.predict(X)
    fpr_rfc, tpr_rfc, threshold_rfc = metrics.roc_curve(y, y_prob)
    auc_rfc = metrics.auc(fpr_rfc, tpr_rfc)
    score_rfc = metrics.accuracy_score(y, y_pred)
    print([score_rfc, auc_rfc])


def auc_f(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def bp():
    X_train, X_test, y_train, y_test = get_data()
    input_size = X_train.shape[1]
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_size * 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', auc_f])
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=20)
    score = model.evaluate(X_test, y_test, batch_size=10)
    print(score)
    # 实测
    X, y = get_test_data()
    score = model.evaluate(X, y, batch_size=10)
    print(score)


def lstm():
    X_train, X_test, y_train, y_test = get_data()
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    y_train = y_train.values.reshape((y_train.shape[0], 1, 1))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_test = y_test.values.reshape((y_test.shape[0], 1, 1))
    model = Sequential()
    model.add(LSTM(X_train.shape[2], input_shape=(1, X_train.shape[2]),
                   activation='relu',
                   return_sequences=True))
    for i in range(2):
        model.add(LSTM(output_dim=32 * (i + 1),
                       activation='relu',
                       return_sequences=True))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc_f])
    model.fit(X_train, y_train,
              epochs=30,
              batch_size=10)
    score = model.evaluate(X_test, y_test, batch_size=10)
    print(score)
    # 实测
    X, y = get_test_data()
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))
    y = y.values.reshape((y.shape[0], 1, 1))
    score = model.evaluate(X, y, batch_size=10)
    print(score)

def lstm_tuning():
    df = pd.read_csv("/Users/alanp/Downloads/bishedata/hot/train.csv", header=-1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = MinMaxScaler().fit_transform(X)
    print(X.shape)
    kf = StratifiedKFold(n_splits=10)
    k = 0
    for index_train, index_test in kf.split(X, y):
        k += 1
        print("##### Fold " + str(k))
        X_train, X_test = X[index_train], X[index_test]
        y_train, y_test = y.iloc[index_train], y.iloc[index_test]

        # rfc = RandomForestClassifier()
        # rfc.fit(X_train, y_train)
        # y_prob = rfc.predict_proba(X_test)[:, 1]
        # y_pred = rfc.predict(X_test)
        # fpr_rfc, tpr_rfc, threshold_rfc = metrics.roc_curve(y_test, y_prob)
        # auc_rfc = metrics.auc(fpr_rfc, tpr_rfc)
        # score_rfc = metrics.accuracy_score(y_test, y_pred)
        # print([score_rfc, auc_rfc])

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        y_train = y_train.values.reshape((y_train.shape[0], 1, 1))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_test = y_test.values.reshape((y_test.shape[0], 1, 1))
        model = Sequential()
        model.add(LSTM(X_train.shape[2], input_shape=(1, X_train.shape[2]),
                       activation='relu',
                       return_sequences=True))
        for i in range(10):
            model.add(LSTM(50,
                           activation='relu',
                           return_sequences=True))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc_f])
        model.fit(X_train, y_train,
                  epochs=5,
                  batch_size=10,
                  verbose=0)
        score = model.evaluate(X_test, y_test, batch_size=10)
        print(score)


def lstm_example():
    max_features = 20000
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc_f])

    print('Train...')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=15, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(score)


if __name__ == "__main__":
    # logisticRegression()
    # svm()
    # knn()
    # decision_tree()
    # random_forest()
    # bp()
    # lstm()

    # lstm_tuning()
    # lstm_example()

    random_forest_tuning()
