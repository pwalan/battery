# coding:utf-8
import codecs
import csv

output_file = "/Users/alanp/Projects/AbleCloud/data/test.csv"
csvfile = open(output_file, 'w',encoding='utf-8-sig')
writer = csv.writer(csvfile)
result = "ID, 开始时间, 结束时间, 时长, 开始SOC, 结束SOC, 状态, 最高温度, 最低温度, 最大电流, 最小电流, 最大电压, 最小电压, 电量"
writer.writerow(['ID', '开始时间', '结束时间'])
csvfile.close()
