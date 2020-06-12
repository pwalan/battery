# coding:utf-8
import csv
import datetime
import pymysql

input_file = "/Users/alanp/Downloads/bishedata/shortcircuit/so.csv"
output_dir = "/Users/alanp/Downloads/bishedata/shortcircuit/origin/"

db = pymysql.connect("localhost", "root", "pwalan", "battery")
cursor = db.cursor()

with open(input_file) as input:
    reader = csv.reader(input)
    i = 387
    for row in reader:
        i += 1
        print(str(i) + ' ' + row[0])
        time_end = datetime.datetime.strptime(row[0], '%Y/%m/%d %H:%M') - datetime.timedelta(minutes=40)
        time_start = time_end - datetime.timedelta(minutes=30)
        # hot
        # sql = "SELECT * FROM hot_data WHERE vid=" + str(row[0]) + " AND time >= '" + str(time_start) + "' AND time <='" + str(time_end)+"'"
        # uc
        sql = "SELECT * FROM uc_data WHERE get_time >= '" + str(time_start) + "' AND get_time <='" + str(time_end)+"'"
        print(sql)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            with open(output_dir + 'us' + str(i) + ".csv", 'w') as f:
                write = csv.writer(f)
                write.writerows(results)
        except:
            print("Error: unable to fetch data")
