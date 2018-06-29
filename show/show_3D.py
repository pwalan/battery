import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_path = "/Users/alanp/Projects/AbleCloud/data/10001737_output.csv"
datas = pd.read_csv(input_path, encoding="utf-8-sig")

X1 = []
Y1 = []
Z1 = []
X2 = []
Y2 = []
Z2 = []

for data in datas.iterrows():
    if data[1]["state"] == 0:
        # 放电
        X1.append(data[1]["start_SOC"])
        Y1.append(data[1]["end_SOC"]-data[1]["start_SOC"])
        Z1.append(data[1]["Q"])
    else:
        # 充电
        X2.append(data[1]["start_SOC"])
        Y2.append(data[1]["end_SOC"]-data[1]["start_SOC"])
        Z2.append(data[1]["Q"])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, Y1, Z1, c='r', marker='o', label='DisCharge')
ax.scatter(X2, Y2, Z2, c='b', marker='^', label='Charge')
ax.set_xlabel('start_SOC')
ax.set_ylabel('ΔSOC')
ax.set_zlabel('Q')
plt.legend()
plt.show()
