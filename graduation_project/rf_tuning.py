# coding:utf-8

import csv
from random import seed
from random import randrange
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sko.PSO import PSO
from sklearn import metrics
from math import sqrt
import random
import pandas as pd
from sklearn.metrics import roc_auc_score


# 加载数据
def loadCSV(filename):
    dataSet = []
    with open(filename, 'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet


# 除了判别列，其他列都转换为float类型
def column_to_float(dataSet):
    featLen = len(dataSet[0]) - 1
    for data in dataSet:
        for column in range(featLen):
            data[column] = float(data[column].strip())


# 将数据集分成N块，方便交叉验证
def spiltDataSet(dataSet, n_folds):
    fold_size = int(len(dataSet) / n_folds)
    dataSet_copy = list(dataSet)
    dataSet_spilt = []
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:  # 这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
            index = randrange(len(dataSet_copy) - 1)  # 有放回的随机采样
            fold.append(dataSet_copy.pop(index))  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        dataSet_spilt.append(fold)
    return dataSet_spilt


# 构造数据子集
def get_subsample(dataSet, ratio):
    subdataSet = []
    lenSubdata = round(len(dataSet) * ratio)
    while len(subdataSet) < lenSubdata:
        index = randrange(len(dataSet) - 1)  # 有放回的随机采样
        subdataSet.append(dataSet[index])
    # print len(subdataSet)
    return subdataSet


# 分割数据集
def data_spilt(dataSet, index, value):
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# 计算分割代价
def spilt_loss(left, right, class_values):
    loss = 0.0
    for class_value in class_values:
        left_size = len(left)
        if left_size != 0:  # 防止除数为零
            prop = [row[-1] for row in left].count(class_value) / float(left_size)
            loss += (prop * (1.0 - prop))
        right_size = len(right)
        if right_size != 0:
            prop = [row[-1] for row in right].count(class_value) / float(right_size)
            loss += (prop * (1.0 - prop))
    return loss


# 获取特征重要性
def get_feature_importance(dataSet):
    df = pd.DataFrame(dataSet)
    rfc = RandomForestClassifier(max_depth=15, n_estimators=15, min_samples_leaf=1, min_samples_split=2)
    rfc.fit(df.iloc[:, :-1], df.iloc[:, -1])
    arr = np.argsort(np.array(rfc.feature_importances_))
    return arr[:-int(0.4 * len(dataSet[0]))]


# 选取任意的n个特征，在这n个特征中，选取分割时的最优特征
def get_best_spilt(dataSet, n_features):
    features = []
    class_values = list(set(row[-1] for row in dataSet))
    b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None
    # 特征选取优化：如果选取的特征的重要性排名在所有特征低于k%，则重新选
    flag = True
    while flag:
        features = []
        while len(features) < n_features:
            index = randrange(len(dataSet[0]) - 1)
            if index not in features:
                features.append(index)
        if set(features).issubset(get_feature_importance(dataSet)):
            flag = True
        else:
            flag = False
    # print 'features:',features
    for index in features:
        for row in dataSet:
            left, right = data_spilt(dataSet, index, row[index])
            loss = spilt_loss(left, right, class_values)
            if loss < b_loss:
                b_index, b_value, b_loss, b_left, b_right = index, row[index], loss, left, right
    # print b_loss
    # print type(b_index)
    return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}


# 决定输出标签
def decide_label(data):
    output = [row[-1] for row in data]
    return max(set(output), key=output.count)


# 子分割，不断地构建叶节点的过程
def sub_spilt(root, n_features, max_depth, min_size, depth):
    left = root['left']
    # print left
    right = root['right']
    del (root['left'])
    del (root['right'])
    # print depth
    if not left or not right:
        root['left'] = root['right'] = decide_label(left + right)
        # print 'testing'
        return
    if depth > max_depth:
        root['left'] = decide_label(left)
        root['right'] = decide_label(right)
        return
    if len(left) < min_size:
        root['left'] = decide_label(left)
    else:
        root['left'] = get_best_spilt(left, n_features)
        # print 'testing_left'
        sub_spilt(root['left'], n_features, max_depth, min_size, depth + 1)
    if len(right) < min_size:
        root['right'] = decide_label(right)
    else:
        root['right'] = get_best_spilt(right, n_features)
        # print 'testing_right'
        sub_spilt(root['right'], n_features, max_depth, min_size, depth + 1)


# 构造决策树
def build_tree(dataSet, n_features, max_depth, min_size):
    root = get_best_spilt(dataSet, n_features)
    sub_spilt(root, n_features, max_depth, min_size, 1)
    return root


# 预测测试集结果
def predict(tree, row):
    predictions = []
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']


# predictions=set(predictions)


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    res = max(set(predictions), key=predictions.count)
    return res


# 创建随机森林
def random_forest(train, test, ratio, n_feature, max_depth, min_size, n_trees):
    trees = []
    for i in range(n_trees):
        train = get_subsample(train, ratio)
        tree = build_tree(train, n_feature, max_depth, min_size)
        # print 'tree %d: '%i,tree
        trees.append(tree)
    # predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]
    return predict_values


# 计算准确率
def accuracy(predict_values, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == int(predict_values[i]):
            correct += 1
    return correct / float(len(actual))

# 计算AUC
def auc_f(train_set, test_set, actual):
    df_train = pd.DataFrame(train_set)
    df_test = pd.DataFrame(test_set)
    rfc = RandomForestClassifier()
    rfc.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
    prob = list(rfc.predict_proba(df_test.iloc[:, :-1])[:, 1])
    auc = roc_auc_score(np.array(actual), np.array(prob))
    return auc

def training():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    seed(1)
    dataSet = loadCSV('/Users/alanp/Downloads/bishedata/resistance/train.csv')
    column_to_float(dataSet)
    n_folds = 10
    max_depth = 15
    min_size = 1
    ratio = 1.0
    # n_features=sqrt(len(dataSet)-1)
    n_features = 9
    n_trees = 15
    folds = spiltDataSet(dataSet, n_folds)
    scores = []

    # for fold in folds:
    #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #     train_set = folds[:]
    #     train_set.remove(fold)
    #     # print len(folds)
    #     train_set = sum(train_set, [])  # 将多个fold列表组合成一个train_set列表
    #     # print len(train_set)
    #     test_set = []
    #     for row in fold:
    #         row_copy = list(row)
    #         # row_copy[-1] = None
    #         test_set.append(row_copy)
    #     # for row in test_set:
    #     # print row[-1]
    #     actual = [int(row[-1]) for row in fold]
    #     predict_values = random_forest(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)
    #     accur = accuracy(predict_values, actual)
    #     auc = auc_f(train_set, test_set, actual)
    #     scores.append([accur, auc])

    fold = folds[-2]
    train_set = folds[
                :]  # 此处不能简单地用train_set=folds，这样用属于引用,那么当train_set的值改变的时候，folds的值也会改变，所以要用复制的形式。（L[:]）能够复制序列，D.copy() 能够复制字典，list能够生成拷贝 list(L)
    train_set.remove(fold)
    # print len(folds)
    train_set = sum(train_set, [])  # 将多个fold列表组合成一个train_set列表
    # print len(train_set)
    test_set = []
    for row in fold:
        row_copy = list(row)
        # row_copy[-1] = None
        test_set.append(row_copy)
    # for row in test_set:
    # print row[-1]
    actual = [int(row[-1]) for row in fold]
    predict_values = random_forest(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)
    accur = accuracy(predict_values, actual)
    auc = auc_f(train_set, test_set, actual)
    scores.append([accur, auc])

    print('Trees is %d' % n_trees)
    print('scores:%s' % scores)
    # print('mean score:%s' % (sum(scores) / float(len(scores))))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


# 参数调优目标函数
def params_tuning_function(x):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    max_depth, min_size, n_features, n_trees = x
    max_depth = int(max_depth)
    min_size = int(min_size)
    n_features = int(n_features)
    n_trees = int(n_trees)
    seed(1)
    dataSet = loadCSV('/Users/alanp/Downloads/bishedata/hot/train.csv')
    column_to_float(dataSet)
    n_folds = 5
    ratio = 1.0
    folds = spiltDataSet(dataSet, n_folds)
    fold = folds[n_folds - 2]
    train_set = folds[:]
    train_set.remove(fold)
    train_set = sum(train_set, [])
    test_set = []
    for row in fold:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    # for row in test_set:
    # print row[-1]
    actual = [row[-1] for row in fold]
    predict_values = random_forest(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)
    accur = accuracy(predict_values, actual)
    print(max_depth, min_size, n_features, n_trees)
    print('accuracy:%s' % accur)
    return 1 - accur

# 参数调优过程
def params_tuning():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    pso = PSO(func=params_tuning_function, dim=4, pop=100, max_iter=50, lb=[5, 1, 3, 5], ub=[30, 5, 15, 30], w=0.8,
              c1=0.5,
              c2=0.5)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    training()
    # params_tuning()
