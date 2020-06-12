# coding:utf-8
import time
import csv

from pso.ipso import pso
from pso.kpso import kpso
from pso.ga import ga


def format2timestamp(raw_time):
    import time
    from datetime import datetime
    c = datetime.strptime(raw_time.strip(), '%Y-%m-%d %H:%M')
    nat_time = int(time.mktime(c.timetuple())) + c.microsecond / 1000000.00
    return nat_time


filename = '/Users/alanp/Downloads/param/mgl10.csv'

datas = []

# 从csv文件中读取数据
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)

    for row in reader:
        try:
            datas.append([row[0], float(row[1]), float(row[2]), float(row[3])])
        except:
            print("Data Error")

inputs = []
if len(datas) > 10:
    cur_soc = datas[0][1]
    inputs.append({'soc': cur_soc, 't': [], 'voltage': [], 'current': [],'date':[]})
    i = 0
    for data in datas:
        # print(data)
        tmp = int(int(data[1]) / 10)
        if data[1] == cur_soc:
            date = data[0].replace('/', '-')
            inputs[i]['t'].append(format2timestamp(date))
            inputs[i]['voltage'].append(data[2])
            inputs[i]['current'].append(data[3])
            inputs[i]['date'].append(date)
        else:
            cur_soc = data[1]
            i += 1
            inputs.append({'soc': cur_soc, 't': [], 'voltage': [], 'current': [],'date':[]})

now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
filename = '/Users/alanp/Downloads/param/' + now+".csv"
f = open(filename, 'a')
start_time = time.clock()
for input in inputs:
    # print(input)
    if (len(input['t']) > 0):
        best_solution=pso(input)
        print(input['date'][0])
        print(best_solution)
        f.write(str(input['date'][0])+","+str(best_solution[0])+","+str(best_solution[1])+","+str(best_solution[2])+","+str(best_solution[3])+ "\n")
f.close()
print('总计算耗时：', time.clock() - start_time, "s")
