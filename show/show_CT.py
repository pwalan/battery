import os
import pandas as pd
import matplotlib.pyplot as plt

# 展示电流随时间的变化

b_id = "B0006"
root_path = "/Users/alanp/Downloads/battery/"
root_dir = root_path + b_id
# 获取该目录下所有文件（即，每次充放电信息表）
csv_lists = os.listdir(root_dir)
csv_lists.remove(".DS_Store")
# 根据充放电编号排序
csv_lists.sort(key=lambda s: int(s.split("_")[1]))
k = 0
# 要展示的充电数据的序号（忽略放电数据后，如2表示第2次充电数据）
show_num = [2, 10, 30, 50, 80, 100, 120, 140, 160]
# 存储读取到的数据
datas = []
# 读取数据
for i in range(len(csv_lists)):
    input_path = os.path.join(root_dir, csv_lists[i])
    if os.path.isfile(input_path) & ("discharge" not in csv_lists[i]) & ("charge" in csv_lists[i]):
        k += 1
        if k in show_num:
            datas.append(pd.read_csv(input_path, header=1))
# 画线
for i in range(len(datas)):
    plt.plot(datas[i].iloc[:, 5], datas[i].iloc[:, 1], label=str(show_num[i]))
    plt.legend()
plt.xlabel('time')
plt.ylabel('current')
plt.show()
