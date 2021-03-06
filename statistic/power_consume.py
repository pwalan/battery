# -*- coding:utf-8 -*-
import time
import pymysql
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdate
import json
import uuid
import sys

def main(params):
    # print(params)
    return_dict={}
    return_dict['message']=''
    # json_str = json.dumps(params)
    params = json.loads(params)
    picName="日耗电量"+str(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
    # picurl = "F:\\area\\" + picName + ".png"
    # csvurl = "F:\\area\\" + picName + ".csv"
    picurl = "/home/picture/" + picName + ".png"
    csvurl="/home/csv/"+picName+".csv"
    # f = open(csvurl, 'a',encoding='utf-8-sig')
    #print(params)
    early_start = ' 07:00:00'
    early_end = ' 09:00:00'
    late_start = ' 17:00:00'
    late_end = ' 19:00:00'
    flag=True
    db = pymysql.connect("10.103.244.129", "root", "yang1290", "baas")
    cursor = db.cursor()

    vid = int(params['vehicleId'])
    start_time = params['startTime']
    end_time = params['endTime']
    dates = []
    start_time = datetime.strptime(start_time, "%Y-%m-%d")
    end_time = datetime.strptime(end_time, "%Y-%m-%d")
    while start_time <= end_time:
        date_str = start_time.strftime("%Y-%m-%d")
        dates.append(date_str)
        start_time += timedelta(days=1)

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
            #print(result1[0][0])
            #print(len(result1))
            if result1[0][0]!=None:
                #print("noooo")
                flag=False
            cursor.execute(sql_mileage_early)
            result2 = cursor.fetchall()
        except:
            return_dict['picurl'] = ""
            return_dict['csvurl'] = ""
            return_dict['code'] = 1
            return_dict['message'] +=date+ " "
            json_return = json.dumps(return_dict, ensure_ascii=False)
        if result2[0][0] is not None and result2[0][0] - result2[0][1] != 0:
            early.append(result1[0][0] * 6 / 3600 / (result2[0][0] - result2[0][1])/1000)
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
            if result1[0][0]!=None:
                flag=False
            cursor.execute(sql_mileage_late)
            result2 = cursor.fetchall()
        except:
            return_dict['picurl'] = ""
            return_dict['csvurl'] = ""
            return_dict['code'] = 1
            return_dict['message'] += date+" "
            json_return = json.dumps(return_dict, ensure_ascii=False)
        if result2[0][0] is not None and result2[0][0] - result2[0][1] != 0:
            late.append(result1[0][0] * 6 / 3600 / (result2[0][0] - result2[0][1])/1000)
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
            if result1[0][0]!=None:
                flag=False
            cursor.execute(sql_mileage_usual)
            result2 = cursor.fetchall()
        except:
            return_dict['picurl'] = ""
            return_dict['csvurl'] = ""
            return_dict['code'] = 1
            return_dict['message'] +=date+ " "
            json_return = json.dumps(return_dict, ensure_ascii=False)
        if result2[0][0] is not None and result2[0][0] - result2[0][1] != 0:
            usual.append(result1[0][0] * 6 / 3600 / (result2[0][0] - result2[0][1])/1000)
        else:
            usual.append(0)
        if flag:
            return_dict['picurl'] = ""
            return_dict['csvurl'] = ""
            return_dict['code'] = 1
            return_dict['message'] +=date+ " "
            json_return = json.dumps(return_dict, ensure_ascii=False)


    # f.write("车辆编号: " + str(vid) + "\n")
    # f.write("日期," + str(dates).replace("[", "").replace("]", "") + "\n")
    # f.write("早高峰" + str(early).replace("[", "").replace("]", "") + "\n")
    # f.write("晚高峰" + str(late).replace("[", "").replace("]", "") + "\n")
    # f.write("平时" + str(usual).replace("[", "").replace("]", ""))
    # f.close()

    x = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

    plt.figure()
    plt.plot(x, early, 'o-', label='早高峰')
    plt.plot(x, late, '*-', label='晚高峰')
    plt.plot(x, usual, 's-', label='平时')
    plt.xlabel("日期")
    plt.ylabel("耗电量（kWh/km）")
    plt.title('电池编号 ' + str(vid) + ' 的日耗电量')
    ax = plt.gca()  # 表明设置图片的各个轴，plt.gcf()表示图片本身
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 横坐标标签显示的日期格式
    plt.xticks(pd.date_range(x[0], x[-1], freq='1D'))  # 设置x轴时间间隔
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig(picurl)

    return_dict['picurl'] = picName + '.png'
    return_dict['csvurl'] = picName + '.csv'
    if return_dict['message']=='':
        return_dict['code'] = 0
        return_dict['message'] = '成功'
    else:
        return_dict['code'] = 1
        return_dict['message']="车辆编号"+str(vid)+"在日期"+return_dict['message'].rstrip()+"数据为空"
    json_return = json.dumps(return_dict, ensure_ascii=False)
    return json_return


if __name__ == '__main__':
    # params = sys.argv[1]
    params = '{"startTime":"2019-08-13","endTime":"2019-08-15","vehicleId":"1"}'
    #print(params)
    res = main(params)
    print(res)
