# coding:utf-8
import csv
import datetime
import pandas as pd


def hot():
    input_dir = "/Users/alanp/Downloads/bishedata/hot/nh/nh"

    out = open('/Users/alanp/Downloads/bishedata/hot/nhtrain.csv', 'w')
    csv_write = csv.writer(out, dialect='excel')

    for i in range(521):
        try:
            df = pd.read_csv(input_dir + str(i) + ".csv", header=-1)
            max_data = df.max()
            min_data = df.min()
            data = []
            data.append(max_data[3])
            data.append(min_data[3])
            data.append(max_data[4])
            data.append(min_data[4])
            data.append(max_data[5])
            data.append(min_data[5])
            data.append(max_data[5] - min_data[5])
            data.append(max_data[6])
            data.append(min_data[6])
            data.append(max_data[7])
            data.append(min_data[7])
            data.append(max_data[10])
            data.append(min_data[10])
            data.append(max_data[12])
            data.append(min_data[12])
            data.append((max_data[10] - min_data[10]) / 0.5)
            data.append((max_data[12] - min_data[12]) / 0.5)
            data.append(max_data[14])
            data.append(min_data[14])
            data.append(0)
            csv_write.writerow(data)
        except:
            print("No Such File")


def unconsistency():
    input_dir = "/Users/alanp/Downloads/bishedata/shortcircuit/us/us"

    out = open('/Users/alanp/Downloads/bishedata/shortcircuit/ustrain.csv', 'w')
    csv_write = csv.writer(out, dialect='excel')
    for i in range(517):
        try:
            df = pd.read_csv(input_dir + str(i) + ".csv", header=-1)
            max_data = df.max()
            min_data = df.min()
            data = []
            data.append(max_data[9])
            data.append(min_data[9])
            data.append(max_data[8])
            data.append(min_data[8])
            data.append(max_data[7])
            data.append(min_data[7])
            data.append(max_data[7] - min_data[7])
            data.append(max_data[4] - min_data[4])
            data.append(max_data[5] - min_data[5])
            data.append(max_data[4])
            data.append(min_data[4])
            data.append(max_data[5])
            data.append(min_data[5])
            data.append(max_data[13])
            data.append(min_data[13])
            data.append(max_data[3])
            data.append(min_data[3])
            data.append(0)
            print(data)
            csv_write.writerow(data)
        except:
            print("Data Error!")


if __name__ == '__main__':
    # hot()
    unconsistency()
