# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/msyh.ttf')


def get_validation_ac():
    hot = (0.7786, 0.8244, 0.7405, 0.8321, 0.8702, 0.7634, 0.8015)
    uc = (0.7222, 0.7847, 0.6806, 0.7847, 0.8194, 0.6667, 0.7153)
    re = (0.6735, 0.6871, 0.6667, 0.7347, 0.7687, 0.6871, 0.6939)
    sc = (0.6154, 0.6538, 0.6385, 0.6846, 0.7231, 0.6462, 0.6677)
    return hot, uc, re, sc


def get_validation_auc():
    hot = (0.6824, 0.9121, 0.6918, 0.8381, 0.9150, 0.5411, 0.5021)
    uc = (0.6287, 0.8737, 0.6791, 0.7463, 0.9076, 0.5000, 0.5224)
    re = (0.6018, 0.7508, 0.6200, 0.6902, 0.8063, 0.4992, 0.5436)
    sc = (0.6482, 0.7424, 0.6180, 0.6650, 0.7217, 0.5023, 0.5204)
    return hot, uc, re, sc


def get_test_ac():
    hot = (0.7397, 0.8219, 0.7123, 0.9178, 0.9041, 0.6849, 0.6939)
    uc = (0.6835, 0.7468, 0.6329, 0.7594, 0.8354, 0.6329, 0.6431)
    re = (0.7000, 0.7100, 0.7222, 0.7444, 0.7667, 0.6667, 0.6778)
    sc = (0.7067, 0.7467, 0.6400, 0.7067, 0.7333, 0.6667, 0.6800)
    return hot, uc, re, sc


def get_test_auc():
    hot = (0.7426, 0.8891, 0.7565, 0.9252, 0.9626, 0.5418, 0.5029)
    uc = (0.5966, 0.8545, 0.6555, 0.7593, 0.8586, 0.5000, 0.5226)
    re = (0.5244, 0.7522, 0.7136, 0.6917, 0.7356, 0.4993, 0.5183)
    sc = (0.6248, 0.7792, 0.5172, 0.6700, 0.7804, 0.5022, 0.5205)
    return hot, uc, re, sc


def show(data, ylable, pattern=''):
    n_groups = 7
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
           label="热失控")
    ax.bar(index + bar_width, uc, bar_width,
           alpha=opacity, color='y',
           edgecolor='y',
           error_kw=error_config,
           hatch=pattern,
           label="不一致")
    ax.bar(index + 2 * bar_width, re, bar_width,
           alpha=opacity, color='b',
           edgecolor='b',
           error_kw=error_config,
           hatch=pattern,
           label="内阻异常")
    ax.bar(index + 3 * bar_width, sc, bar_width,
           alpha=opacity, color='g',
           edgecolor='g',
           error_kw=error_config,
           hatch=pattern,
           label="内短路")
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(('LR', 'SVM', 'KNN', 'DT', 'RF', 'BP', "LSTM"))
    ax.legend(prop=getChineseFont(), loc='upper right')
    plt.xlabel('Algorithms')
    plt.ylabel(ylable)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    fig.tight_layout()
    plt.savefig('/Users/alanp/Desktop/result.png', dpi=360)
    # plt.show()


def get_hot2_ac():
    hot1_vac = (0.7786, 0.8244, 0.7405, 0.8321, 0.8702, 0.7634, 0.8015)
    hot1_tac = (0.7397, 0.8219, 0.7123, 0.9178, 0.9041, 0.6849, 0.6939)
    hot2_vac = (0.7786, 0.8244, 0.7328, 0.8244, 0.8345, 0.7023, 0.7328)
    hot2_tac = (0.7397, 0.8356, 0.6986, 0.8767, 0.8868, 0.6849, 0.6936)
    return hot1_vac, hot1_tac, hot2_vac, hot2_tac


def get_hot2_auc():
    hot1_vauc = (0.6824, 0.9121, 0.6918, 0.8381, 0.9150, 0.5411, 0.5021)
    hot1_tauc = (0.7426, 0.8891, 0.7565, 0.9252, 0.9626, 0.5418, 0.5029)
    hot2_vauc = (0.6838, 0.8879, 0.6883, 0.7608, 0.8842, 0.4976, 0.5176)
    hot2_tauc = (0.7391, 0.8808, 0.7408, 0.8713, 0.9556, 0.5029, 0.5181)
    return hot1_vauc, hot1_tauc, hot2_vauc, hot2_tauc


def show2(data, ylable, pattern=''):
    n_groups = 7
    plt.style.use("ggplot")
    hot1_v, hot1_t, hot2_v, hot2_t = data
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8
    error_config = {'ecolor': '0.3'}
    ax.bar(index, hot1_v, bar_width,
           alpha=opacity, color='r',
           edgecolor='r',
           error_kw=error_config,
           hatch=pattern,
           label="验证集1")
    ax.bar(index + bar_width, hot1_t, bar_width,
           alpha=opacity, color='tomato',
           edgecolor='tomato',
           error_kw=error_config,
           hatch=pattern,
           label="测试集1")
    ax.bar(index + bar_width + bar_width, hot2_v, bar_width,
           alpha=opacity, color='b',
           edgecolor='b',
           error_kw=error_config,
           hatch=pattern,
           label="验证集2")
    ax.bar(index + bar_width + bar_width + bar_width, hot2_t, bar_width,
           alpha=opacity, color='deepskyblue',
           edgecolor='deepskyblue',
           error_kw=error_config,
           hatch=pattern,
           label="测试集2")
    ax.set_xticks(index + 4 * bar_width / 4)
    ax.set_xticklabels(('LR', 'SVM', 'KNN', 'DT', 'RF', 'BP', "LSTM"))
    ax.legend(prop=getChineseFont())
    plt.xlabel('Algorithms')
    plt.ylabel(ylable)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    fig.tight_layout()
    plt.savefig('/Users/alanp/Desktop/result.png', dpi=360)
    # plt.show()


def get_rf_ac():
    hot = (0.8702, 0.9091, 0.9041, 0.9041)
    uc = (0.8194, 0.8611, 0.8354, 0.8101)
    re = (0.7687, 0.8648, 0.7667, 0.8222)
    sc = (0.7231, 0.7846, 0.7333, 0.7467)
    return hot, uc, re, sc


def get_rf_auc():
    hot = (0.9150, 0.9571, 0.9626, 0.9730)
    uc = (0.9076, 0.9063, 0.8586, 0.9097)
    re = (0.8063, 0.9200, 0.7356, 0.8531)
    sc = (0.7217, 0.8505, 0.7804, 0.7920)
    return hot, uc, re, sc


def show3(data, ylable, pattern=''):
    n_groups = 4
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
           label="热失控")
    ax.bar(index + bar_width, uc, bar_width,
           alpha=opacity, color='y',
           edgecolor='y',
           error_kw=error_config,
           hatch=pattern,
           label="不一致")
    ax.bar(index + 2 * bar_width, re, bar_width,
           alpha=opacity, color='b',
           edgecolor='b',
           error_kw=error_config,
           hatch=pattern,
           label="内阻异常")
    ax.bar(index + 3 * bar_width, sc, bar_width,
           alpha=opacity, color='g',
           edgecolor='g',
           error_kw=error_config,
           hatch=pattern,
           label="内短路")
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(('Validation_RF', 'Validation_PSO-RF', 'Test_RF', 'Test_PSO-RF'))
    ax.legend(prop=getChineseFont(), loc='lower left')
    plt.xlabel('DataSet_Algorithms')
    plt.ylabel(ylable)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    fig.tight_layout()
    plt.savefig('/Users/alanp/Desktop/result.png', dpi=360)
    # plt.show()

def get_rf_ac2():
    hot = (0.8702, 0.8345, 0.9041, 0.8868)
    uc = (0.8194, 0.7917, 0.8354, 0.8101)
    re = (0.7687, 0.7297, 0.7667, 0.6889)
    sc = (0.7231, 0.6923, 0.7333, 0.7067)
    return hot, uc, re, sc


def get_rf_auc2():
    hot = (0.9150, 0.8842, 0.9626, 0.9556)
    uc = (0.9076, 0.8691, 0.8586, 0.8876)
    re = (0.8063, 0.7880, 0.7356, 0.7667)
    sc = (0.7217, 0.7132, 0.7804, 0.7656)
    return hot, uc, re, sc


def show4(data, ylable, pattern=''):
    n_groups = 4
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
           label="热失控")
    ax.bar(index + bar_width, uc, bar_width,
           alpha=opacity, color='y',
           edgecolor='y',
           error_kw=error_config,
           hatch=pattern,
           label="不一致")
    ax.bar(index + 2 * bar_width, re, bar_width,
           alpha=opacity, color='b',
           edgecolor='b',
           error_kw=error_config,
           hatch=pattern,
           label="内阻异常")
    ax.bar(index + 3 * bar_width, sc, bar_width,
           alpha=opacity, color='g',
           edgecolor='g',
           error_kw=error_config,
           hatch=pattern,
           label="内短路")
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(('Validation1', 'Validation2', 'Test1', 'Test2'))
    ax.legend(prop=getChineseFont(), loc='lower left')
    plt.xlabel('DataSet')
    plt.ylabel(ylable)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    fig.tight_layout()
    plt.savefig('/Users/alanp/Desktop/result.png', dpi=360)
    # plt.show()


if __name__ == "__main__":
    # show(get_validation_ac(), 'Accuracy')
    # show(get_validation_auc(), 'AUC', '//')
    # show(get_test_ac(), 'Accuracy')
    # show(get_test_auc(), 'AUC', '//')

    # show2(get_hot2_ac(), 'Accuracy')
    # show2(get_hot2_auc(), 'AUC', '//')

    # show3(get_rf_ac(), 'Accuracy')
    # show3(get_rf_auc(), 'AUC', '//')

    # show4(get_rf_ac2(), 'Accuracy')
    show4(get_rf_auc2(), 'AUC', '//')
