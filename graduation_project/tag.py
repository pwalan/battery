# coding:utf-8

import csv
import pandas as pd
import numpy as np


def unconsistency():
    df = pd.read_csv("/Users/alanp/Projects/201806_高比功率电池项目/数据/盟固利/北邮数据/北邮数据10-201801-201812.csv", header=-1)
    print(df.head())
    print(df.shape)
    out = open('/Users/alanp/Downloads/bishedata/unconsistency/uc.csv', 'w')
    csv_write = csv.writer(out, dialect='excel')
    count = 0
    for index, row in df.iterrows():
        try:
            t = [int(x) for x in row[9].split("|")]
            u = [float(x) for x in row[10].split("|")]
            v = a = np.std(u) + (np.max(u) - np.min(u)) + np.std(t) * 0.01 + (np.max(t) - np.min(t)) * 0.001
            if v > 0.3:
                print(row[0])
                csv_write.writerow(str(row[0]))
                count += 1
        except:
            print("error")
    print(count)


if __name__ == '__main__':
    unconsistency()
