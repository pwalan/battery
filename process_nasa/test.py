# coding:utf-8
import codecs
import csv
import numpy as np
import math
from dateutil import parser
import matplotlib.pyplot as plt

# output_file = "/Users/alanp/Projects/AbleCloud/data/test.csv"
# csvfile = open(output_file, 'w',encoding='utf-8-sig')
# writer = csv.writer(csvfile)
# result = "ID, 开始时间, 结束时间, 时长, 开始SOC, 结束SOC, 状态, 最高温度, 最低温度, 最大电流, 最小电流, 最大电压, 最小电压, 电量"
# writer.writerow(['ID', '开始时间', '结束时间'])
# csvfile.close()

# for i in np.arange(0, 1, 0.1):
#     print(i)

# a = np.array([1, 2, 3, 4, 5, 6, 7])
# b = np.array([0, 2, 3, 4, 5, 6, 7])
# # c = math.sqrt(np.sum(((a - b) / a) ** 2))
# c = np.append(a, b)
# print(c)

# start_time = parser.parse('20171116161033')
# end_time=parser.parse('20171116170903')
# print((end_time-start_time).seconds)

# a = [1, 2, 3, 4, 5]
# b = [6, 7, 8, 9]
# print(a + b)



x = np.linspace(0, 2 * np.pi, 50)
wave=np.cos(x)
transformed=np.fft.fft(wave)

plt.subplot(1, 2, 1)
plt.plot(wave)

plt.subplot(1, 2, 2)
plt.plot(transformed)

plt.show()
