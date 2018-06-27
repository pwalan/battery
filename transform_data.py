# coding:utf-8
import csv
import time
import datetime

# 将10001737原始数据转为充放电数据

def format2timestamp(raw_time):
    import time
    from datetime import datetime
    c = datetime.strptime(raw_time.strip(), '%Y-%m-%d %H:%M:%S')
    nat_time = int(time.mktime(c.timetuple())) + c.microsecond / 1000000.00
    return nat_time


input_file = "/Users/alanp/Projects/AbleCloud/data/10001737.csv"
output_file = "/Users/alanp/Projects/AbleCloud/data/10001737_output.csv"

# 统计过程中的各个量
ischarge = False  # 是否是充电状态
isfirst = True  # 是否是第一次状态转换
isFirstData = True  # 是否是总的第一条数据
isP = True  # 之前是否是正电流
thisStartTime = ""  # 放电开始时间
thisStartSOC = 0  # 放电开始SOC
countN = 0  # 负电流次数
countP = 0  # 正电流次数
negCurrentStartTime = ""  # 充电开始时间
negCurrentStartSOC = 0  # 充电开始SOC
lastTime = ""  # 上一条数据的时间
lastSOC = ""  # 上一条数据的SOC
maxTemperature = 0  # 最高温度
minTemperature = 100  # 最低温度
maxCurrent = 0.0  # 最大电流
minCurrent = 1000.0  # 最小电流
maxVol = 0.0  # 最大电压
minVol = 1000.0  # 最小电压
timeformate = "%Y/%m/%d %H:%M:%S"
pvalue = []  # 正电流首次出现时的数据
nvalue = []  # 负电流首次出现时的数据
COUNT = 5  # 判断充放需要的电正负电流次数
TIMELIMIT = 30  # 时间间隔阈值，放电过程中前后两条数据的时间差大于阈值要切分数据
It_sum = 0.0  # 充放电电量总和

output = open(output_file, 'w', encoding='utf-8-sig')
result = u"ID, 开始时间, 结束时间, 时长, 开始SOC, 结束SOC, 状态, 最高温度, 最低温度, 最大电流, 最小电流, 最大电压, 最小电压, 电量\n"
print(result)
output.write(result)
with open(input_file) as input:
    reader = csv.reader(input)
    for row in reader:
        if len(row) >= 25:
            x = []
            x.append(row[0])  # ID
            x.append(row[20])  # 日期
            x.append(row[2])  # 总电流（字符串）
            x.append(int(row[3]))  # SOC
            x.append(int(row[12]))  # 最高温度
            x.append(int(row[15]))  # 最低温度
            x.append(float(row[2]))  # 总电流
            x.append(float(row[1]))  # 总电压
            # 开始统计
            # 判断温度大小
            if x[4] >= maxTemperature:
                maxTemperature = x[4]
            if x[5] < minTemperature:
                minTemperature = x[5]
            # 判断电流大小
            if x[6] >= maxCurrent:
                maxCurrent = x[6]
            if x[6] < minCurrent:
                minCurrent = x[6]
            # 判断电压大小
            if x[7] >= maxVol:
                maxVol = x[7]
            if x[7] < minVol:
                minVol = x[7]

            It_sum += abs(x[6]) * 30 / 3600
            if isFirstData:
                lastTime = x[1]
                lastSOC = x[3]
            # 判断时间是否断档，如果断档需要重新开始
            date1 = time.strptime(lastTime, timeformate)
            date2 = time.strptime(row[20], timeformate)
            date1 = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
            date2 = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
            delta = date2 - date1
            interval = (delta.days * 24 * 3600 + delta.seconds) / 60
            if interval <= TIMELIMIT:
                # 数据无断档，进行判断
                # 判断电流正负
                if x[2].__contains__("-"):
                    # 电流为负
                    if countN < COUNT:
                        # 如果负电流次数小于COUNT，表明此时不一定在充电，需要接着判断
                        countN += 1
                        countP -= 1
                        if isFirstData:
                            isFirstData = False
                            nvalue = x
                            maxTemperature = x[4]
                            minTemperature = x[5]
                            maxCurrent = x[6]
                            minCurrent = x[6]
                            maxVol = x[7]
                            minVol = x[7]
                            isP = False
                        if isP:
                            # 如果之前是正电流，出现一个负电流可能就是充电开始或是刹车开始
                            nvalue = x
                            isP = False
                    else:
                        # 如果负电流次数大于COUNT，而且之前是放电状态，需要转为充电状态
                        if isfirst:
                            # 第一次状态的标记
                            negCurrentStartTime = nvalue[1]
                            negCurrentStartSOC = nvalue[3]
                            isfirst = False
                            ischarge = True
                        if not isfirst and not ischarge:
                            negCurrentStartTime = nvalue[1]
                            negCurrentStartSOC = nvalue[3]
                            date1 = time.strptime(thisStartTime, timeformate)
                            date2 = time.strptime(negCurrentStartTime, timeformate)
                            date1 = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
                            date2 = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
                            delta = date2 - date1
                            duration = (delta.days * 24 * 3600 + delta.seconds)
                            inputs = {'soc': thisStartSOC, 't': [], 'voltage': [], 'current': []}
                            # discharge
                            result = (x[0] + "," + thisStartTime + "," + negCurrentStartTime + "," + str(duration) + ","
                                      + str(thisStartSOC) + "," + str(negCurrentStartSOC) + "," + str(0) + "," + str(
                                        maxTemperature) + "," + str(minTemperature) +
                                      "," + str(maxCurrent) + "," + str(minCurrent) + "," + str(maxVol) + "," + str(
                                        minVol)) + "," + str(It_sum) + "\n"
                            # 将结果写入csv
                            output.write(result)
                            print(result)
                            It_sum = 0.0
                            maxTemperature = nvalue[4]
                            minTemperature = nvalue[5]
                            maxCurrent = nvalue[6]
                            minCurrent = nvalue[6]
                            maxVol = nvalue[7]
                            minVol = nvalue[7]
                        ischarge = True
                else:
                    # 电流为正或0
                    if countP < COUNT:
                        # 如果正电流次数小于COUNT，表明此时不一定在放电，需要接着判断
                        countP += 1
                        countN -= 1
                        if isFirstData:
                            isFirstData = False
                            pvalue = x
                            maxTemperature = x[4]
                            minTemperature = x[5]
                            maxCurrent = x[6]
                            minCurrent = x[6]
                            maxVol = x[7]
                            minVol = x[7]
                            isP = True
                        if not isP:
                            pvalue = x
                            isP = True
                    else:
                        if isfirst:
                            # 第一次状态的标记
                            thisStartTime = pvalue[1]
                            thisStartSOC = pvalue[3]
                            isfirst = False
                            ischarge = False
                        if not isfirst and ischarge:
                            thisStartTime = pvalue[1]
                            thisStartSOC = pvalue[3]
                            date1 = time.strptime(negCurrentStartTime, timeformate)
                            date2 = time.strptime(thisStartTime, timeformate)
                            date1 = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
                            date2 = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
                            delta = date2 - date1
                            duration = (delta.days * 24 * 3600 + delta.seconds)
                            # charge
                            result = (x[0] + "," + negCurrentStartTime + "," + thisStartTime + "," + str(duration) + ","
                                      + str(negCurrentStartSOC) + "," + str(thisStartSOC) + "," + str(1) + "," + str(
                                        maxTemperature) + "," + str(minTemperature) +
                                      "," + str(maxCurrent) + "," + str(minCurrent) + "," + str(maxVol) + "," + str(
                                        minVol)) + "," + str(It_sum) + "\n"
                            # 将结果写入csv
                            output.write(result)
                            It_sum = 0.0
                            print(result)
                            maxTemperature = pvalue[4]
                            minTemperature = pvalue[5]
                            maxCurrent = pvalue[6]
                            minCurrent = pvalue[6]
                            maxVol = pvalue[7]
                            minVol = pvalue[7]
                        ischarge = False
            else:
                # 数据断档，重新开始
                if ischarge and countN >= COUNT:
                    date1 = time.strptime(negCurrentStartTime, timeformate)
                    date2 = time.strptime(lastTime, timeformate)
                    date1 = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
                    date2 = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
                    delta = date2 - date1
                    duration = (delta.days * 24 * 3600 + delta.seconds)
                    # charge
                    result = (x[0] + "," + negCurrentStartTime + "," + lastTime + "," + str(duration) + ","
                              + str(negCurrentStartSOC) + "," + str(lastSOC) + "," + str(1) + "," + str(
                                maxTemperature) + "," + str(minTemperature) +
                              "," + str(maxCurrent) + "," + str(minCurrent) + "," + str(maxVol) + "," + str(
                                minVol)) + "," + str(It_sum) + "\n"
                    # 将结果写入csv
                    output.write(result)
                    print(result)
                elif not ischarge and countP >= COUNT:
                    date1 = time.strptime(thisStartTime, timeformate)
                    date2 = time.strptime(lastTime, timeformate)
                    date1 = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
                    date2 = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
                    delta = date2 - date1
                    duration = (delta.days * 24 * 3600 + delta.seconds)
                    # discharge
                    result = (x[0] + "," + thisStartTime + "," + lastTime + "," + str(duration) + ","
                              + str(thisStartSOC) + "," + str(lastSOC) + "," + str(0) + "," + str(
                                maxTemperature) + "," + str(minTemperature) +
                              "," + str(maxCurrent) + "," + str(minCurrent) + "," + str(maxVol) + "," + str(
                                minVol)) + "," + str(It_sum) + "\n"
                    # 将结果写入csv
                    output.write(result)
                    print(result)

                isfirst = True
                It_sum = 0.0
                countN = 0
                countP = 0
                maxTemperature = x[4]
                minTemperature = x[5]
                maxCurrent = x[6]
                minCurrent = x[6]
                maxVol = x[7]
                minVol = x[7]
                if x[2].__contains__("-"):
                    nvalue = x
                    negCurrentStartTime = x[1]
                    negCurrentStartSOC = x[3]
                    isP = False
                else:
                    pvalue = x
                    thisStartTime = pvalue[1]
                    thisStartSOC = pvalue[3]
                    isP = True
                pso_input_datas = []
            lastTime = x[1]
            lastSOC = x[3]

output.close()
