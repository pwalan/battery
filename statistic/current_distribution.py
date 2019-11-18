# -*- coding:utf-8 -*-

import pymysql
import matplotlib.pyplot as plt
import json
import uuid


def main(params):
    print(params)
    json_str = json.dumps(params)
    params = json.loads(json_str)
    pic_url = "/home/pic/" + str(uuid.uuid1()) + ".jpg"

    db = pymysql.connect("10.103.244.129", "root", "yang1290", "baas")
    cursor = db.cursor()

    vid = params['vehicleId']
    dates = params['dates'].split('|')
    start_time = ' 00:00:00'
    end_time = ' 23:59:59'
    charge = []
    discharge=[]
    for date in dates:
        sql = "SELECT Current FROM driving_log WHERE vehicle_id=%d AND time >= '%s' AND time <= '%s'" % (
            vid, date + start_time, date + end_time)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                if row[0]>=0:
                    discharge.append(int(row[0]))
                else:
                    charge.append(int(row[0]))
        except:
            print("Error: unable to fetch data")

    margin=1
    fig = plt.figure()
    # 充电电流分布
    plt.subplot(2,1,1)
    bins = range(-350, 0, 10)
    plt.xlim(-350, 0)
    plt.title("Vehicle No."+str(vid)+" Charge Count-distribution")
    plt.xlabel('Current')
    plt.ylabel('Count')
    prob, left, rectangle = plt.hist(x=charge, bins=bins, normed=False, histtype='bar', color=['r'])
    for x, y in zip(left, prob):
        plt.text(x + 10 / 2, y , '%d' % y, ha='center', va='bottom')

    # 放电点流分布
    plt.subplot(2, 1, 2)
    bins = range(0, 350, 10)
    plt.xlim(0, 350)
    plt.title("Vehicle No."+str(vid)+"DisCharge Count-distribution")
    plt.xlabel('Current')
    plt.ylabel('Count')
    prob, left, rectangle = plt.hist(x=discharge, bins=bins, normed=False, histtype='bar', color=['blue'])
    for x, y in zip(left, prob):
        # 频次分布数据 normed=False
        plt.text(x + 10 / 2, y, '%d' % y, ha='center', va='bottom')

    fig.tight_layout()
    fig.set_dpi(150)
    plt.show()
    # plt.savefig(pic_url)
    return pic_url


if __name__ == '__main__':
    # params = sys.argv[1]
    params = {'dates': '2019-08-13', 'vehicleId': 1}
    pic_url = main(params)
    print(pic_url)
