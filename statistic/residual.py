import matplotlib.pyplot as plt
import random

data = []
for i in range(1000):
    soh = 0.9
    rul = 0.9
    if i <= 200:
        soh = random.uniform(0.9, 1.0)
        rul = random.uniform(0.9, 1.0)
    elif i <= 400:
        soh = random.uniform(0.8, 0.9)
        rul = random.uniform(0.8, 0.9)
    elif i <= 600:
        soh = random.uniform(0.7, 0.8)
        rul = random.uniform(0.7, 0.8)
    elif i <= 800:
        soh = random.uniform(0.6, 0.7)
        rul = random.uniform(0.6, 0.7)
    else:
        soh = random.uniform(0.4, 0.6)
        rul = random.uniform(0.4, 0.6)
    # rul = random.uniform(0.1, 0.9)
    co = random.uniform(0.4, 0.8)
    # co = 0.8
    s = random.uniform(0.4, 0.8)
    # s = 0.8
    dc = random.uniform(0.4, 0.8)
    # dc = 0.8
    ch = random.uniform(0.4, 0.8)
    # ch = 0.8
    h = random.uniform(0.4, 0.8)
    # h = 0.8
    # data.append(soh * rul * co * s * dc * ch * h)
    # data.append(pow(soh*rul*co*s*dc*ch*h,1.0/7.0))
    data.append(pow(pow(soh, 1.0 / 3.0) * pow(rul, 1 / 3.0) * pow(co, 1.0 / 2.0) * pow(s, 1.0 / 2.0) * pow(dc, 1.0 / 1.0) * pow(ch,1.0 / 1.0) * pow(h, 1.0 / 1.0), 1.0 / 7.0))

plt.scatter(range(1000), data)
plt.show()
