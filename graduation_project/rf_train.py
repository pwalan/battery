# coding:utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as K


def get_data():
    df = pd.read_csv("/Users/alanp/Downloads/bishedata/hot/train.csv", header=-1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def get_test_data():
    df = pd.read_csv("/Users/alanp/Downloads/bishedata/hot/test.csv", header=-1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # X = MinMaxScaler().fit_transform(X)
    return X, y


def random_forest():
    X_train, X_test, y_train, y_test = get_data()
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
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

def auc_f(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


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

if __name__ == "__main__":
    # 结果的前一个值是准确率，后一个值是AUC
    random_forest()
    # 结果的第一个值是loss，第二个值是准确率，最后一个值是AUC
    # lstm()
