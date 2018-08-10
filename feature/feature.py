import os
import csv


def get_duration():
    b_id = "B0006"
    root_path = "/Users/alanp/Downloads/"
    # 根目录，此目录下包含编号为B00xx的电池的每次充放电详细信息。（每次充电或放电均为一张excel表格，记录每一时刻电池信息）
    root_dir = root_path + b_id
    lists = os.listdir(root_dir)  # 获取该目录下所有文件（即，每次充放电信息表）

    # 在桌面新建以电池编号命名的csv文件，用于存储此电池每次充放电的信息（原本每次充放电信息为一张excel表格，此处将一张excel表格总结为一条记录
    # 每条记录包括：（电池编号、充放电次数、充电or放电状态、持续时间、本次充or放电过程中的最大电池电压、最小电池电压、最大电池电流、最小电池电流、
    # 最大温度、最小温度、最大电源电流、最小电源电流、最大电源电压、最小电源电压、soh）
    with open(root_path + b_id + ".csv", "w") as csv_file2:
        writer2 = csv.writer(csv_file2)

        # 写表头
        writer2.writerow(
            ["id", "count", "state", "last_time", "cc_duration", "cv_duration", "slope", "vertical", "soh"])

        # 逐文件进行读取（此处的文件为：root_dir目录下的文件，即，每个文件对应一张excel表格，包含一次充电or放电的详细信息）
        for i in range(0, len(lists)):

            input_path = os.path.join(root_dir, lists[i])

            # 打印当前读取的文件名
            print(input_path)

            if os.path.isfile(input_path) & ("discharge" not in lists[i]) & ("charge" in lists[i]):

                # 打开此文件
                with open(input_path, 'r') as csv_file:
                    reader = csv.reader(csv_file)

                    sum_soh = 0.0  # 电池充电or放电总量
                    last_time = 0.0  # 上一行的时间（计算时间差时使用）
                    max_current_charge = 1.5  # 最大电源电流
                    min_voltage_charge = 4.0  # 最小电源电压
                    is_first = -1  # 计数，用于跳过前两行记录
                    final_time = 0.0  # 充or放电的持续时间
                    cc_duration = 0.0  # 恒流充电时间
                    cv_duration = 0.0  # 恒压充电时间
                    cc_duration_final = 0.0  # 恒流充电结束的时间点
                    first_time = 0.0  # 开始充电的时间
                    flag = True
                    voltage_measured = 0.0  # 电池电压
                    last_voltage = 0.0  # 上次电压
                    slope = 0.0
                    voltage = []
                    count = 0
                    all_time = []
                    vertical = 0.0
                    vertical_time_start = 0
                    vertical_time_over = 0
                    vertical_voltage_start = 0.0
                    vertical_voltage_over = 0.0

                    # 逐行读取数据
                    for row in reader:

                        is_first += 1

                        # 判断如果是前两行数据的话 就continue
                        if is_first < 2:
                            last_time = float(row[5])
                            last_voltage = float(row[0])
                            if is_first == 1:
                                first_time = last_time
                            continue

                        # 原始数据的第5列为电源电压
                        voltage_charge = float(row[4])

                        # TODO 获取恒流充电的第一个拐点
                        d_voltage = float(row[0]) - last_voltage
                        if d_voltage <= 0.001 and d_voltage != 0:
                            vertical=-()

                        # 当电源电压小于1时，证明充or放电结束，不去处理
                        if voltage_charge > min_voltage_charge:

                            # 当前电源电流，用于计算cc_duration
                            current_charge = float(row[3])

                            # 当前电池电流，用于计算soh
                            current = float(row[1])

                            voltage_measured = float(row[0])
                            voltage.append(voltage_measured)

                            # 此时的时间
                            time = float(row[5])
                            all_time.append(time)
                            if time < 303:
                                vertical_time_start = time
                                vertical_voltage_start = voltage_measured
                            if time < 586:
                                vertical_time_over = time
                                vertical_voltage_over = voltage_measured

                            # 最后一次以1.5A充电
                            if current_charge > max_current_charge:
                                # flag = False
                                cc_duration = time - first_time
                                cc_duration_final = time
                                count = is_first - 2

                            # 计算充电or放电过程中，电池总容量，用电池电流*（此时时间 - 上次时间）/3600
                            sum_soh += abs(current) * (time - last_time) / 3600

                            # 将这次的时间赋值给last_time
                            last_time = time


                        else:
                            # 当电源电压<1时，进入此分支，此时充电or放电已经结束，跳出循环，上一条数据的时间即为总充电or放电时间
                            break

                    # 将信息写入csv文件  list[i]为文件名。
                    slope = (voltage[count] - voltage[count - 60]) / (all_time[count] - all_time[count - 60])
                    vertical_before = (vertical_voltage_over - vertical_voltage_start) / (
                            vertical_time_over - vertical_time_start)
                    final_time = last_time - first_time
                    cv_duration = last_time - cc_duration_final
                    battery_id = lists[i].split('_')[0]
                    count = lists[i].split('_')[1]
                    if int(count) <= 180:
                        vertical = -1 / vertical_before
                    else:
                        vertical = -1 / vertical_before
                    state = lists[i].split('_')[2]
                    soh = sum_soh / 2.0

                    writer2.writerow(
                        [battery_id, count, state, final_time, cc_duration, cv_duration, slope, vertical, soh])


if __name__ == '__main__':
    get_duration()
