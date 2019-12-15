import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def probability_distribution(data, bins_interval=1, margin=1):
    bins = range(min(data), max(data) + bins_interval - 1, bins_interval)
    print(len(bins))
    for i in range(0, len(bins)):
        print(bins[i])
    plt.xlim(min(data) - margin, max(data) + margin)
    plt.title("probability-distribution")
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    plt.hist(x=data, bins=bins, histtype='bar', color=['r'])
    plt.show()


def show_distributon(data):
    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    data = [1, 4, 6, 7, 8, 9, 11, 11, 12, 12, 13, 13, 16, 17, 18, 22, 25]
    # probability_distribution(data=data, bins_interval=5, margin=0)
    # show_distributon(data)
    print(pow(0.8, 1.0 / 5.0))
