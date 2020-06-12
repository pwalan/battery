# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


def get_rf_ac():
    hot = (0, 0.7538, 0, 0.7769, 0)
    uc = (0, 0.7042, 0, 0.7323, 0)
    re = (0, 0.7260, 0, 0.7534, 0)
    sc = (0, 0.6719, 0, 0.7187, 0)
    return hot, uc, re, sc


def get_rf_auc():
    hot = (0, 0.8757, 0, 0.8973, 0)
    uc = (0, 0.8239, 0, 0.8411, 0)
    re = (0, 0.7124, 0, 0.7753, 0)
    sc = (0, 0.7566, 0, 0.7869, 0)
    return hot, uc, re, sc


def show(data, ylable, pattern=''):
    n_groups = 5
    plt.style.use("ggplot")
    hot, uc, re, sc = data
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.7
    error_config = {'ecolor': '0.4'}
    ax.bar(index, hot, bar_width,
           alpha=opacity, color='r',
           edgecolor='r',
           error_kw=error_config,
           hatch=pattern,
           label="Thermal Runaway")
    ax.bar(index + bar_width, uc, bar_width,
           alpha=opacity, color='y',
           edgecolor='y',
           error_kw=error_config,
           hatch=pattern,
           label="Inconsistent")
    ax.bar(index + 2 * bar_width, re, bar_width,
           alpha=opacity, color='b',
           edgecolor='b',
           error_kw=error_config,
           hatch=pattern,
           label="Resistance Abnormal")
    ax.bar(index + 3 * bar_width, sc, bar_width,
           alpha=opacity, color='g',
           edgecolor='g',
           error_kw=error_config,
           hatch=pattern,
           label="Internal Short-circuit")
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(('', 'RF', '', 'IRF', ''))
    ax.legend(loc='lower left')
    plt.xlabel('Algorithm')
    plt.ylabel(ylable)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fig.tight_layout()
    plt.savefig('/Users/alanp/Desktop/result.png', dpi=360)


if __name__ == "__main__":
    # show(get_rf_ac(), 'Accuracy')
    show(get_rf_auc(), 'AUC', '//')
