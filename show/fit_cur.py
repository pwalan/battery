import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_path = '/Users/alanp/Downloads/B0006/B0006_3_charge_24.csv'
output_path = '/Users/alanp/Downloads/tmp.csv'
# 读入数据
data = pd.read_csv(input_path, header=1,
                   names=['v', 'i', 'c', 'i1', 'v1', 't'])
# 找到恒流充电的结束
indexs = data[data['v'] > 4.2].index
# 截取出恒流充电的数据
X = data['t'][:indexs[0]].values
Y = data['v'][:indexs[0]].values
# 使用9次多项式拟合曲线
z = np.polyfit(X, Y, 9)
# 求拟合后曲线上各时间点对应的电压值
Yvals = np.polyval(z, X)

# 求曲线上各点的一阶和二阶导
D_1 = np.poly1d(z).deriv(m=1)
Dvals1 = np.polyval(D_1, X)
D_2 = np.poly1d(z).deriv(m=2)
Dvals2 = np.polyval(D_2, X)
# 写入csv
tmp = pd.DataFrame({'x': X, 'd1': Dvals1, 'd2': Dvals2})
tmp.to_csv(output_path)

# 绘制拟合的曲线和实际数据
plt.scatter(X, Y, label='real data')
plt.plot(X, Yvals, c='black', linestyle="--", label='fitting curve')
plt.legend()
plt.show()
