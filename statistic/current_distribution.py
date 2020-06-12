# -*- coding:utf-8 -*-

import pymysql
import matplotlib.pyplot as plt
import json
import uuid
import math


def get_rmes(datas):
    if len(datas) == 0:
        return 0
    sum = 0
    for data in datas:
        sum += data * data
    return math.sqrt(sum / len(datas))


def main(params):
    print(params)
    json_str = json.dumps(params)
    params = json.loads(json_str)
    picurl = "/home/pic/" + str(uuid.uuid1()) + ".jpg"
    # TODO 改地址
    csvurl = "/Users/alanp/Downloads/param/" + "电流概率分布" + str(uuid.uuid1()) + ".csv"
    # csvurl="/home/csv/"+str(uuid.uuid1())+".csv"
    f = open(csvurl, 'a', encoding='utf-8-sig')

    db = pymysql.connect("10.103.244.129", "root", "yang1290", "baas")
    cursor = db.cursor()

    vid = params['vehicleId']
    f.write("车辆编号：" + str(vid) + "\n")
    start_date = params['startTime']
    end_date = params['endTime']
    start_time = ' 00:00:00'
    end_time = ' 23:59:59'
    charge = []
    discharge = []

    sql = "SELECT Current FROM driving_log WHERE vehicle_id=%d AND time >= '%s' AND time <= '%s'" % (
        vid, start_date + start_time, end_date + end_time)
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            if row[0] >= 5:
                discharge.append(int(row[0]))
            elif row[0] <= -5:
                charge.append(int(row[0]))
    except:
        print("Error: unable to fetch data")

    rmes_charge = -get_rmes(charge)
    rmes_discharge = get_rmes(discharge)

    frequency = []
    rates = []
    currents = []

    print()
    # f.write("电流," + str(list(range(-350, 360, 10))).replace("[", "").replace("]", "") + "\n")
    margin = 1
    fig = plt.figure()
    # 充电电流分布
    plt.subplot(2, 1, 1)
    plt.grid()
    bins = range(-350, 10, 10)
    plt.xlim(-350, 0)
    plt.title("车辆编号为 " + str(vid) + " 的充电电流分布 " + str(rmes_charge))
    plt.xlabel('电流')
    plt.ylabel('采集点')
    prob, left, rectangle = plt.hist(x=charge, bins=bins, normed=False, histtype='bar', color=['r'])
    prob1, left1, rectangle1 = plt.hist(x=charge, bins=bins, normed=True, histtype='bar', color=['r'])
    # f.write("频次," + str(list(prob)).replace("[", "").replace("]", ""))
    currents.extend(list(bins))
    frequency.extend(prob)
    rates.extend(prob1)
    for x, y in zip(left, prob):
        plt.text(x + 10 / 2, y, '%d' % y, ha='center', va='bottom')

    # 放电点流分布
    plt.subplot(2, 1, 2)
    plt.grid()
    bins = range(0, 360, 10)
    plt.xlim(0, 350)
    plt.title("车辆编号为 " + str(vid) + " 的放电电流分布 " + str(rmes_discharge))
    plt.xlabel('电流')
    plt.ylabel('采集点')
    prob, left, rectangle = plt.hist(x=discharge, bins=bins, normed=False, histtype='bar', color=['blue'])
    prob2, left2, rectangle2 = plt.hist(x=discharge, bins=bins, normed=True, histtype='bar', color=['blue'])
    # f.write(str(list(prob)).replace("[", "").replace("]", "") + "\n")
    currents.extend(list(bins))
    frequency.extend(prob)
    rates.extend(prob2)
    for x, y in zip(left, prob):
        # 频次分布数据 normed=False
        plt.text(x + 10 / 2, y, '%d' % y, ha='center', va='bottom')

    fig.tight_layout()
    fig.set_dpi(150)
    # TODO 取消show，开启保存
    plt.show()
    # plt.savefig(picurl)

    # f.write("频率," + str(list(prob1)).replace("[", "").replace("]", "") + str(list(prob2)).replace("[", "").replace("]", ""))
    f.write("均方根值" + "," + str(rmes_charge) + "," + str(rmes_discharge)+"\n")
    f.write("电流, 频次, 频率\n")
    for i in range(len(currents)):
        try:
            f.write(str(currents[i]) + "," + str(frequency[i]) + "," + str(rates[i]) + "\n")
        except:
            continue
    f.close()
    return "{\"picurl\":\"" + str(picurl) + "\",\"csvurl\":\"" + str(csvurl) + "\",\"code\":\"0\",\"message\":\"成功\"}"


if __name__ == '__main__':
    # params = sys.argv[1] '{"startTime": "2019-11-07", "endTime": "2019-11-08", "vehicleId": "1"}'
    params = {'startTime': '2019-11-07', 'endTime': '2019-11-08', 'vehicleId': 1}
    res = main(params)
    print(res)
