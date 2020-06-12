# -*- coding:utf-8 -*-
import sys
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import socket
import pymysql
import re
import datetime

# 模型存放的路径
model_dir = '/Users/alanp/Desktop/'


def hot_feature(arr):
    df = pd.DataFrame(arr)
    print(df.head())
    max_data = df.max()
    min_data = df.min()
    data = []
    data.append(float(max_data[9]))
    data.append(float(min_data[9]))
    data.append(float(max_data[12]))
    data.append(float(min_data[12]))
    data.append(float(max_data[15]))
    data.append(float(min_data[15]))
    data.append(int(max_data[15]) - int(min_data[15]))
    data.append(float(max_data[5]))
    data.append(float(min_data[5]))
    data.append(float(max_data[7]))
    data.append(float(min_data[7]))
    data.append(float(max_data[13]))
    data.append(float(min_data[13]))
    data.append(float(max_data[14]))
    data.append(float(min_data[14]))
    data.append((float(max_data[13]) - float(min_data[13])) / 0.5)
    data.append((float(max_data[14]) - float(min_data[14])) / 0.5)
    data.append(float(max_data[5]) - float(min_data[5]))
    data.append(float(max_data[7]) - float(min_data[7]))
    return data


def other_feature(arr):
    df = pd.DataFrame(arr)
    max_data = df.max()
    min_data = df.min()
    data = []
    data.append(float(max_data[9]))
    data.append(float(min_data[9]))
    data.append(float(max_data[12]))
    data.append(float(min_data[12]))
    data.append(float(max_data[15]))
    data.append(float(min_data[15]))
    data.append(int(max_data[15]) - int(min_data[15]))
    data.append(float(max_data[5]))
    data.append(float(min_data[5]))
    data.append(float(max_data[7]))
    data.append(float(min_data[7]))
    data.append(float(max_data[13]))
    data.append(float(min_data[13]))
    data.append(float(max_data[14]))
    data.append(float(min_data[14]))
    data.append(float(max_data[5]) - float(min_data[5]))
    data.append(float(max_data[7]) - float(min_data[7]))
    return data


def predict(arr, type, port):
    model = joblib.load(model_dir + type + '.m')
    testx = []
    if type == 'hot':
        testx = np.asarray(hot_feature(arr)).reshape(1, 19)
    else:
        testx = np.asarray(other_feature(arr)).reshape(1, 17)
    res = model.predict(testx)

    conn = pymysql.connect(host='10.103.244.129', user="root", passwd="yang1290", db="baas", port=3306,
                           charset="utf8")
    cur = conn.cursor()
    if len(res) == 0:
        return 0
    if type == 'hot':
        model_id = 0
    if type == 'resistance':
        model_id = 1
    if type == 'shortcircuit':
        model_id = 2
    if type == 'unconsistency':
        model_id = 3

    sql = """insert into monitor_result(vehicle_id, data_time, is_read, result, model_id, port_id)
                           values (%d,str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s'),0,%f,%d,%d)""" % (
        int(arr[0][1]), arr[-1][2], res[0], model_id, int(port))
    cur.execute(sql)
    conn.commit()
    return res[0]


def main(port, type='hot'):
    arr = []
    i = 1
    j = 1
    while (1):
        str_port = int(port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('10.103.244.129', int(port)))
        print('Client model1 : model1 start!')
        sock.send(bytes("model1\n", encoding='utf-8'))
        data = sock.recv(204800)
        if len(data) < 20:
            continue
        b = data[1:-2]
        b = b.decode()
        frame = re.split(r",(?![^(]*\))", b)
        print(frame)
        arr.append(frame)
        if i == 100:
            predict(arr, type, port)
        if i > 100:
            j += 1
        if j == 20:
            j = 0
            arr = arr[-100:]
            predict(arr, type, port)
        i += 1
        sock.close()


if __name__ == '__main__':
    port = 8081
    type = 'hot'
    # 'hot'代表热失控，model_id=0
    # 'resistance'代表内阻异常，model_id=1
    # 'shortcircuit'代表内短路，model_id=2
    # 'unconsistency'代表不一致，model_id=3
    # port = sys.argv[1]
    # type = sys.argv[2]
    main(port, type)
