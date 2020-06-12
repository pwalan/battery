import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib
import time
from datetime import datetime, timedelta
import csv
from sklearn.model_selection import train_test_split
from sko.PSO import PSO


def probability_distribution(data, bins_interval=1, margin=1):
    bins = range(min(data), max(data) + bins_interval - 1, bins_interval)
    print(len(bins))
    for i in range(0, len(bins)):
        print(bins[i])
    plt.xlim(min(data) - margin, max(data) + margin)
    plt.title("probability-distribution")
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    plt.hist(x=data, bins=bins, histtype='bar', color=['r'])
    plt.show()


def show_distributon(data):
    plt.plot(data)
    plt.show()


def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


if __name__ == '__main__':
    data = [1, 4, 6, 7, 8, 9, 11, 11, 12, 12, 13, 13, 16, 17, 18, 22, 25]
    # probability_distribution(data=data, bins_interval=5, margin=0)
    # show_distributon(data)
    # a = "日耗电量"+str(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))

    # time1 = datetime.strptime('2019/9/11 12:25', '%Y/%m/%d %H:%M')
    # time2=time1-timedelta(minutes=30)
    # time3=time2-timedelta(minutes=30)
    # print(str(time2))
    # print(time3)

    # datas = []
    # a = [1, 2, 3, 4, 5]
    # b = ['1', '2', 'jjj']
    # datas.append(a)
    # datas.append(b)
    # print(datas)

    # i = 2
    # try:
    #     df = pd.read_csv("/Users/alanp/Downloads/bishedata/unconsistency/co/co" + str(i) + ".csv", header=-1)
    #     print(df.head())
    #     print(max(df[10][0].split("|")))
    # except:
    #     print("data error")

    # arr = [0.0579168, 0.06664274, 0.07392245, 0.08254095, 0.07092077, 0.06383988,
    #        0.06924479, 0.03665057, 0.04608635, 0.0349087, 0.05093512, 0.1027339,
    #        0.06244015, 0.04342892, 0.03766324, 0.01969933, 0.01817686, 0.03699739,
    #        0.02525102]
    # print(np.argsort(np.array(arr)))
    # arr.sort()
    # print(arr)

    # df = pd.read_csv("/Users/alanp/Downloads/bishedata/hot/train.csv", header=-1)
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # print(pd.concat([X_train, y_train], axis=1))

    # pso = PSO(func=demo_func, dim=3, pop=40, max_iter=150, lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
    # pso.run()
    # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

    # arr = [16, 15, 18, 9, 7, 17, 14, 13, 8, 10, 0, 12, 5, 1, 6, 4, 2, 3, 11]
    # arr1 = arr[:-int(0.3 * len(arr))]
    # brr = [7, 17, 14, 13, 8, 10, 0,11]
    # print(set(brr).issubset(arr1))

    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc = roc_auc_score(y_true, y_scores)
    print(auc)
