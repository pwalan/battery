# -*- coding:utf-8 -*-

import pymysql
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdate
import json
import uuid


def main(params):
    print(params)
    json_str = json.dumps(params)
    params = json.loads(json_str)
    pic_url = "/home/pic/" + str(uuid.uuid1()) + ".jpg"

    early_start = ' 07:00:00'
    early_end = ' 09:00:00'
    late_start = ' 17:00:00'
    late_end = ' 19:00:00'

    db = pymysql.connect("10.103.244.129", "root", "yang1290", "baas")
    cursor = db.cursor()

    vid = params['vehicleId']
    dates = params['dates'].split('|')
    early = []
    late = []
    usual = []

    for date in dates:
        # 早高峰
        sql_power_early = "SELECT SUM(Voltage*Current) FROM driving_log WHERE vehicle_id=%d AND time >='%s' AND time <= '%s' AND Current>=0" % (
            vid, date + early_start, date + early_end)
        sql_mileage_early = "SELECT MAX(GPS_Mileage),MIN(GPS_Mileage) FROM driving_log WHERE vehicle_id=%d AND GPS_Mileage > 0 AND time >='%s' AND time <= '%s' AND Current>=0" % (
            vid, date + early_start, date + early_end)
        try:
            cursor.execute(sql_power_early)
            result1 = cursor.fetchall()

            cursor.execute(sql_mileage_early)
            result2 = cursor.fetchall()
        except:
            print("Error: unable to fetch data")
        if result2[0][0] is not None and result2[0][0] - result2[0][1] != 0:
            early.append(result1[0][0] * 6 / 3600 / (result2[0][0] - result2[0][1]))
        else:
            early.append(0)

        # 晚高峰
        sql_power_late = "SELECT SUM(Voltage*Current) FROM driving_log WHERE vehicle_id=%d AND time >='%s' AND time <= '%s' AND Current>=0" % (
            vid, date + late_start, date + late_end)
        sql_mileage_late = "SELECT MAX(GPS_Mileage),MIN(GPS_Mileage) FROM driving_log WHERE vehicle_id=%d AND GPS_Mileage > 0 AND time >='%s' AND time <= '%s' AND Current>=0" % (
            vid, date + late_start, date + late_end)
        try:
            cursor.execute(sql_power_late)
            result1 = cursor.fetchall()

            cursor.execute(sql_mileage_late)
            result2 = cursor.fetchall()
        except:
            print("Error: unable to fetch data")
        if result2[0][0] is not None and result2[0][0] - result2[0][1] != 0:
            late.append(result1[0][0] * 6 / 3600 / (result2[0][0] - result2[0][1]))
        else:
            late.append(0)

        # 平时
        sql_power_usual = "SELECT SUM(Voltage*Current) FROM driving_log WHERE vehicle_id=%d AND ((time >='%s' AND time <= '%s') OR (time >='%s' AND time <= '%s') OR (time >='%s' AND time <= '%s')) AND Current>=0" % (
            vid, date + ' 00:00:00', date + early_start, date + early_end, date + late_start, date + late_end,
            date + ' 23:59:59')
        sql_mileage_usual = "SELECT MAX(GPS_Mileage),MIN(GPS_Mileage) FROM driving_log WHERE vehicle_id=%d AND GPS_Mileage > 0 AND ((time >='%s' AND time <= '%s') OR (time >='%s' AND time <= '%s') OR (time >='%s' AND time <= '%s')) AND Current>=0" % (
            vid, date + ' 00:00:00', date + early_start, date + early_end, date + late_start, date + late_end,
            date + ' 23:59:59')
        try:
            cursor.execute(sql_power_usual)
            result1 = cursor.fetchall()

            cursor.execute(sql_mileage_usual)
            result2 = cursor.fetchall()
        except:
            print("Error: unable to fetch data")
        if result2[0][0] is not None and result2[0][0] - result2[0][1] != 0:
            usual.append(result1[0][0] * 6 / 3600 / (result2[0][0] - result2[0][1]))
        else:
            usual.append(0)

    print(early)
    print(late)
    print(usual)

    x = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

    plt.figure()
    plt.plot(x, early, 'o-', label='early')
    plt.plot(x, late, '*-', label='late')
    plt.plot(x, usual, 's-', label='usual')
    plt.xlabel("date")
    plt.ylabel("power consumption")
    plt.title('VEHICLE NO.' + str(vid) + ' Power Consumption')
    ax = plt.gca()  # 表明设置图片的各个轴，plt.gcf()表示图片本身
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 横坐标标签显示的日期格式
    plt.xticks(pd.date_range(x[0], x[-1], freq='1D'))  # 设置x轴时间间隔
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.legend()
    plt.show()
    # plt.savefig(pic_url)

    return pic_url


if __name__ == '__main__':
    # params = sys.argv[1]
    params = {'dates': '2019-08-13|2019-08-14|2019-08-15|2019-08-16|2019-08-17', 'vehicleId': 1}
    pic_url = main(params)
    print(pic_url)
