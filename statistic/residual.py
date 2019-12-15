import matplotlib.pyplot as plt
import random

data = []
for i in range(1000):
    soh = random.uniform(0.6, 0.9)
    rul = random.uniform(0.1, 0.9)
    co = random.uniform(0.1, 0.9)
    s = random.uniform(0.1, 0.9)
    dc = random.uniform(0.1, 0.9)
    ch = random.uniform(0.1, 0.9)
    h = random.uniform(0.1, 0.9)
    # data.append(pow(soh*rul*co*s*dc*ch*h,1.0/7.0))
    # data.append(soh*rul*co*s*dc*ch*h)
    data.append(pow(pow(soh, 1.0 / 3.0) * pow(rul, 1 / 3.0) * pow(co, 1.0 / 2.0) * pow(s, 1.0 / 2.0) *
                    pow(dc, 1.0 / 1.0) * pow(ch, 1.0 /1.0) * pow(h, 1.0 / 1.0), 1.0 / 7.0))

plt.plot(data)
plt.show()
