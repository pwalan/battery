# coding:utf-8
# 将种群的自然选择、繁衍互换以及基因突变视为自然过程
# 将对种群的评价单独写进一个函数record_in_God_view中
# 这样无论是在种群初始化过程中，还是代数更迭过程中，都可以该函数实现评价和记录
import random
import math
import copy
import time
import numpy


# def obj_func(x, c):
#     """
#     目标函数
#     :param x:
#     :return:
#     """
#     # f = 10 * math.sin(5 * x[0]) + 7 * math.cos(4 * x[0]) + 5 * math.sin(3 * x[1]) + 8 * math.cos(4 * x[1])
#     f = c[0] * math.sin(5 * x[0]) + 7 * math.cos(4 * x[0]) + c[1] * math.sin(3 * x[1]) + 8 * math.cos(4 * x[1])
#     return f


def obj_func(x, data_4at_SOC):
    if len(data_4at_SOC['voltage']) < 5:
        return -1000
    else:
        u1 = [[] for i in list(range(len(data_4at_SOC['voltage'])))]
        err = [[] for i in list(range(len(u1)))]
        u1[0] = 0
        err[0] = 0
        for i in list(range(len(u1) - 1)):
            Ts = data_4at_SOC['t'][i + 1] - data_4at_SOC['t'][i]
            u1[i + 1] = (-(Ts - 2 * x[2]) * u1[i] + x[1] * Ts * (
                    data_4at_SOC['current'][i + 1] + data_4at_SOC['current'][i])) / (Ts + 2 * x[2])
            err[i + 1] = x[0] - u1[i + 1] - x[3] * (data_4at_SOC['current'][i + 1] + data_4at_SOC['current'][i]) * 0.5 - \
                         data_4at_SOC['voltage'][i + 1]
        obj = 0
        for i in list(range(len(err))):
            obj = obj + math.fabs(err[i])
        return obj


def cal_obj2fit_value(obj_value_set):
    """
    根据目标每个个体的目标函数值计算每个个体的适应度
    适应度函数为“1.2的x次方”，适应度底数越大，种群收敛越快
    :param obj_value_set:
    :return:
    """
    # fit_value_set = [math.exp(obj_value_set[i]) for i in range(len(obj_value_set))]
    fit_value_set = [math.pow(1.05, -obj_value_set[i] / 100) for i in range(len(obj_value_set))]
    return fit_value_set


def ga(input):
    """
    算法入口
    :param input:{'soc': cur_soc, 't': [], 'voltage': [], 'current': []}
    :return:
    """
    start_time = time.clock()
    # 算法参数
    upper_limit = [600, 0.1, 1000, 0.1]  # 参数搜索上界
    lower_limit = [440, 0, 0, 0]  # 参数搜索下届
    #upper_limit = [50, 0.01, 200, 0.001]  # 参数搜索上界
    #lower_limit = [25, 0, 0, 0]  # 参数搜索下届
    pop_size = 200  # 种群大小
    generation_len = 1000  # 种群进化代数
    gene_len = [50, 50, 50, 50]  # 每个染色体上的基因长度
    chrom_n = len(gene_len)  # 染色体数目
    probability_crossover = 0.01  # 交叉互换概率
    probability_mutation = 0.001  # 单个基因片段突变概率
    num_disaster = 10  # 天灾总次数
    count_disaster = 0  # 天灾计数

    # ga_options,pop_ever,record_ever详情见ga_initialization函数中的说明
    [ga_options, pop_ever, record_ever] = ga_initialization(pop_size, generation_len, gene_len, chrom_n, upper_limit,
                                                            lower_limit, probability_crossover, probability_mutation,
                                                            input)
    # 将计算过程存入文件
    now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    filename = '/Users/alanp/Projects/AbleCloud/data/result/' + now + ".csv"
    f = open(filename, 'a')
    f.write("迭代次数,适应度值,计算耗时\n")
    print('=========================================================')
    for i in range(ga_options['generation_len']):
        start_pop_time = time.clock()
        pop_ever.append(selection(pop_ever[-1], record_ever['fit_value_set_ever'][-1], ga_options))
        # 迁出迁入
        pop_ever[-1] = move_in_and_out(pop_ever[-1], ga_options)

        pop_ever[-1] = crossover(pop_ever[-1], ga_options)
        pop_ever[-1] = mutation(pop_ever[-1], record_ever['fit_value_set_ever'][-1], ga_options)

        record_ever = record_in_God_view(pop_ever[-1], record_ever, ga_options)

        # 引入天灾
        if record_ever['pop_perf'][-1][1][2] < 0.0001 and record_ever['pop_perf'][-1][1][
            3] < 0.0001 and count_disaster < num_disaster:
            print("Disaster")
            pop_ever[-1] = disaster(pop_ever[-1], ga_options)
            count_disaster += 1

        print("进化代数：", i)
        print('当代最佳个体：', record_ever['the_best_individual_ever'][-1][0])
        print('当代最佳适应度：', record_ever['the_best_individual_ever'][-1][1])
        print('当代参数解：', record_ever['the_best_solve_ever'][-1])
        print('当代个体均值：[', record_ever['pop_perf'][-1][0][2], ',', record_ever['pop_perf'][-1][0][3], ']')
        print('当代个体标准差：[', record_ever['pop_perf'][-1][1][2], ',', record_ever['pop_perf'][-1][1][3], ']')
        print('当代目标函数最优值：', record_ever['the_best_result_ever'][-1])
        print('整体目标函数最优值：', min(record_ever['the_best_result_ever']))
        print('当代目标函数均值：', record_ever['pop_perf'][-1][0][0])
        print('当代目标函数标准差：', record_ever['pop_perf'][-1][1][0])
        print('计算耗时：', time.clock() - start_pop_time, "s")
        print('=========================================================')
        f.write(str(i+1)+","+str(min(record_ever['the_best_result_ever']))+","+str(time.clock() - start_pop_time)+"\n")

    f.close()
    the_best_in_history_index = break_the_record(record_ever['the_best_individual_ever'])

    print('最佳个体出现代数：', the_best_in_history_index)
    print('最佳个体的染色体：', record_ever['the_best_individual_ever'][the_best_in_history_index][0])
    print('最佳个体的适应度：', record_ever['the_best_individual_ever'][the_best_in_history_index][1])
    print('最佳个体基因解码：', record_ever['the_best_solve_ever'][the_best_in_history_index])
    print('最佳个体的函数值：', record_ever['the_best_result_ever'][the_best_in_history_index])

    print('计算耗时：', time.clock() - start_time, "s")

    return record_ever['the_best_solve_ever'][the_best_in_history_index]


def ga_initialization(pop_size, generation_len, gene_len, chrom_n, upper_limit, lower_limit, probability_crossover,
                      probability_mutation, input):
    """
    初始化
    :param pop_size:
    :param generation_len:
    :param gene_len:
    :param chrom_n:
    :param upper_limit:
    :param lower_limit:
    :param probability_crossover:
    :param probability_mutation:
    :return:
    """
    # 算法参数字典
    # 将前面定义的算法参数存成字典，便于各个函数读取参数
    ga_options = {'pop_size': pop_size, 'generation_len': generation_len, 'gene_len': gene_len,
                  'chrom_n': chrom_n, 'para_upper_limit': upper_limit, 'para_lower_limit': lower_limit,
                  'probability_crossover': probability_crossover,
                  'probability_mutation': probability_mutation, 'input': input}

    # 每一代中出现的种群
    # 是四维数组：第1维表示进化到了第几代，第2维表示该代中所有个体，第3表表示每个个体的染色体，第4维表示染色体上的基因
    pop_ever = []
    pop_1st = [[[random.randint(0, 1) for iii in range(gene_len[ii])] for ii in range(ga_options['chrom_n'])] for i in
               range(pop_size)]
    pop_ever.append(pop_1st)

    # 每一代的记录
    # obj_value_set_ever 二维数组：第1维表示进化到第几代，第2维表示该代中所有个体对应的目标函数值
    # fit_value_set_ever 二维数组：第1维表示进化到第几代，第2维表示该代中所有个体对应的适应度
    # the_best_individual_ever 二维数组：第1维表示进化到第几代，第2维表示该代中最优个体的染色体组成和适应度
    # the_best_solve_ever 二维数组：第1维表示进化到第几代，第2维表示该代中最优个体对应的参数解
    # the_best_result_ever 一维数组：进化到某代中的最优目标函数值
    # pop_perf 三维数组：第1维表示进化到第几代，第2维只有2个数组（[0]均值，[1]标准差），第3维有4个值（[0]目标函数值，[1]适应度，[2]染色体A，[3]染色体B）
    record_ever = {'obj_value_set_ever': [], 'fit_value_set_ever': [], 'the_best_individual_ever': [],
                   'the_best_solve_ever': [], 'the_best_result_ever': [], 'pop_perf': []}
    record_ever = record_in_God_view(pop_ever[-1], record_ever, ga_options)

    return ga_options, pop_ever, record_ever


def record_in_God_view(pop_current, record_ever, ga_options):
    """
    记录当前种群的每个个体的目标函数值、适应度、最优个体、最优解、最优解对应的函数值、每个个体的均值和方差、适应度的均值和方差
    :param pop_current:
    :param record_ever:
    :param ga_options:
    :return:
    """
    obj_value_set, tran_value_set = cal_obj_value(pop_current, ga_options)
    fit_value_set = cal_obj2fit_value(obj_value_set)
    # 从适应度列表中找最大适应度，其索引代表种群中最佳个体
    best_index_in_current_pop = find_the_Best(fit_value_set)

    pop_best = pop_current[best_index_in_current_pop]
    fit_value_best = fit_value_set[best_index_in_current_pop]
    tran_value_best = tran_value_set[best_index_in_current_pop]
    obj_value_best = obj_value_set[best_index_in_current_pop]

    current_pop_perf_mean, current_pop_perf_std = pop_perf(tran_value_set, obj_value_set, fit_value_set)

    record_ever['obj_value_set_ever'].append(obj_value_set)
    record_ever['fit_value_set_ever'].append(fit_value_set)

    record_ever['the_best_individual_ever'].append([pop_best, fit_value_best])
    record_ever['the_best_solve_ever'].append(tran_value_best)
    record_ever['the_best_result_ever'].append(obj_value_best)
    record_ever['pop_perf'].append([current_pop_perf_mean, current_pop_perf_std])

    return record_ever


def cal_obj_value(pop, ga_options):
    """
    计算种群中每个个体对应的目标函数值
    :param pop:
    :param ga_options:
    :return: 每个个体对应的目标函数值、每个个体基因解码后的值
    """
    pop_size = ga_options['pop_size']
    input = ga_options['input']
    obj_value_set = []
    tran_value_set = [[] for i in range(pop_size)]
    for i in range(pop_size):
        # 对个体基因解码
        tran_value_set[i] = decode_chrom(pop[i], ga_options)
        # 计算目标函数值
        obj_value_set.append(obj_func(tran_value_set[i], input))
    return obj_value_set, tran_value_set


def decode_chrom(pop_individual, ga_options):
    """
    根据目标函数参数的上下界确定基因解码值
    :param pop_individual:
    :param ga_options:
    :return:
    """
    upper_limit = ga_options['para_upper_limit']  # 参数搜索上界
    lower_limit = ga_options['para_lower_limit']  # 参数搜索下界
    chrom_n = ga_options['chrom_n']
    gene_len = ga_options['gene_len']

    decode_value = [0 for i in range(chrom_n)]

    for i in range(chrom_n):
        chrom = pop_individual[i]
        for ii in range(gene_len[i]):
            decode_value[i] = decode_value[i] + chrom[ii] * (math.pow(2, ii))
        decode_value[i] = lower_limit[i] + (upper_limit[i] - lower_limit[i]) / math.pow(2, gene_len[i]) * decode_value[
            i]

    return decode_value


def find_the_Best(fit_value_set):
    """
    找出每代中适应度最大的个体
    :param fit_value_set:
    :return:
    """
    best_index = fit_value_set.index(max(fit_value_set))
    return best_index


def pop_perf(tran_value_set, obj_value_set, fit_value_set):
    """
    统计种群中每个个体的均值、方差，适应度的均值和方差
    :param tran_value_set:
    :param obj_value_set:
    :param fit_value_set:
    :return:
    """
    set_mean = list(map(mean_in_GA, [obj_value_set, fit_value_set]))
    set_std = list(map(std_in_GA, [obj_value_set, fit_value_set]))

    if (len(tran_value_set) != 0):
        for i in range(len(tran_value_set[0])):
            set_mean.append(mean_in_GA([x[i] for x in tran_value_set]))
            set_std.append(std_in_GA([x[i] for x in tran_value_set]))

    return set_mean, set_std


def mean_in_GA(x):
    """
    求数组均值
    :param x:
    :return:
    """
    x_array = numpy.array(x)
    x_mean_in_GA = x_array.sum() / len(x)
    return x_mean_in_GA


def std_in_GA(x):
    """
    求数组标准差
    :param x:
    :return:
    """
    x_array = numpy.array(x)
    x_d_array = x_array - x_array.sum() / len(x)
    x_d2_array_deviation = x_d_array * x_d_array
    x_deviation_in_GA = x_d2_array_deviation.sum() / len(x)
    x_std_in_GA = math.sqrt(x_deviation_in_GA)
    return x_std_in_GA


def selection(pop, fit_value_set, ga_options):
    """
    通过轮盘转法选择适应度高的个体
    :param pop:
    :param fit_value_set:
    :param ga_options:
    :return:选择出的新种群（可能出现连续重复的个体）
    """
    cum_fit_value_set = cumsum(fit_value_set)
    sum_fit_value_set = sum(fit_value_set)
    cum_portation_fit_value_set = [cum_fit_value_set[i] / sum_fit_value_set for i in range(len(cum_fit_value_set))]

    pointer = []
    pop_size = ga_options['pop_size']
    for i in range(pop_size):
        # 生成0到1的随机数
        pointer.append(random.random())

    # 对生成的随机数进行排序，方便与累计后的适应度概率进行对比
    pointer.sort()
    pointer_index, fitness_area_index = 0, 0
    new_pop = [[] for i in range(pop_size)]
    while pointer_index < pop_size:
        if pointer[pointer_index] < cum_portation_fit_value_set[fitness_area_index]:
            new_pop[pointer_index] = pop[fitness_area_index]
            pointer_index += 1
        else:
            fitness_area_index += 1
    return new_pop


def cumsum(fit_value_set):
    """
    计算适应度累计值
    :param fit_value_set:
    :return:
    """
    cum_fit_value_set = []
    for i in range(len(fit_value_set)):
        cum_fit_value_set.append(sum(fit_value_set[0:i + 1]))
    return cum_fit_value_set


def crossover(pop, ga_options):
    """
    选择好新种群后进行基因交叉互换，先将个体打乱顺序，然后对种群列表进行顺序处理，
    通过生成一个随机数与交叉互换概率进行比较，一旦小于，则本个体的两条染色体会与下个个体的染色体的一部分进行交换
    :param pop:
    :param ga_options:
    :return:
    """
    pop_size = ga_options['pop_size']
    gene_len = ga_options['gene_len']
    chrom_n = ga_options['chrom_n']
    crossover_probability = ga_options['probability_crossover']

    pop = naturally_flow(pop, ga_options)

    for i in range(pop_size - 1):
        for ii in range(chrom_n):
            if random.random() < crossover_probability:
                length = gene_len[ii]
                cpoint = random.randint(0, length - 1)
                temp1 = []
                temp2 = []
                temp1.extend(pop[i][ii][0:cpoint])
                temp1.extend(pop[i + 1][ii][cpoint:length])
                temp2.extend(pop[i + 1][ii][0:cpoint])
                temp2.extend(pop[i][ii][cpoint:length])
                pop[i][ii] = temp1
                pop[i + 1][ii] = temp2
    return pop


def naturally_flow(pop, ga_options):
    """
    将种群中个体顺序打乱，方便交叉互换
    （因为在前一步通过轮盘算法选出的种群中会连续出现同一个个体）
    :param pop:
    :param ga_options:
    :return:
    """
    pop_size = ga_options['pop_size']
    pop_after_flow = []
    pop1 = copy.deepcopy(pop)
    for i in range(pop_size):
        ii = random.randint(0, pop_size - 1 - i)
        pop_after_flow.append(pop[ii])
        del pop[ii]
    return pop_after_flow


def mutation(pop, fit_value_set, ga_options):
    """
    基因突变
    :param pop:
    :param ga_options:
    :return:
    """
    probability_mutation = ga_options['probability_mutation']
    pop_size = ga_options['pop_size']
    gene_len = ga_options['gene_len']
    chrom_n = ga_options['chrom_n']
    best_index_in_current_pop = find_the_Best(fit_value_set)

    for i in range(pop_size):
        # 对最优个体的突变（中间1位）,100%突变
        if i == best_index_in_current_pop:
            for j in range(chrom_n):
                postion = int(gene_len[j] / 2)
                # if pop[i][j][postion] == 1:
                #     pop[i][j][postion] = 0
                # else:
                #     pop[i][j][postion] = 1
        else:
            for ii in range(chrom_n):
                if random.random() < probability_mutation:
                    length = gene_len[ii]
                    mpoint = random.randint(0, length - 1)
                    if pop[i][ii][mpoint] == 1:
                        pop[i][ii][mpoint] = 0
                    else:
                        pop[i][ii][mpoint] = 1
    return pop


def break_the_record(the_best_ever):
    """
    找到历史上出现全局最优个体的代数
    :param the_best_ever:
    :return:
    """
    the_best_fitness_ever = [the_best_ever[i][1] for i in range(len(the_best_ever))]
    the_best_in_history_index = the_best_fitness_ever.index(max(the_best_fitness_ever))
    return the_best_in_history_index


def disaster(pop, ga_options):
    """
    在种群进化到一定程度后（种群个体标准差趋近于0）引入天灾
    :param pop:
    :param ga_options:
    :return:
    """
    pop_size = ga_options['pop_size']
    gene_len = ga_options['gene_len']
    new_pop = []

    # 策略1 随机选取,只存活10%
    survival_rate = 0.1
    survival_num = int(pop_size * survival_rate)
    for i in range(survival_num):
        new_pop.append(pop[random.randint(0, pop_size - 1)])
    pop_tmp = [[[random.randint(0, 1) for iii in range(gene_len[ii])] for ii in range(ga_options['chrom_n'])] for i in
               range(pop_size - survival_num)]
    new_pop += pop_tmp

    return new_pop


def move_in_and_out(pop, ga_options):
    """
    每一代随机迁出10%，再迁入10%（重新初始化）
    这样天灾基本不会发生
    改变适应度函数对结果影响不大
    :param pop:
    :param ga_options:
    :return:
    """
    pop_size = ga_options['pop_size']
    gene_len = ga_options['gene_len']

    move_rate = 0.1  # 迁入迁出率
    move_num = int(pop_size * move_rate)
    for i in range(move_num):
        del pop[random.randint(0, len(pop) - 1)]

    pop_tmp = [[[random.randint(0, 1) for iii in range(gene_len[ii])] for ii in range(ga_options['chrom_n'])] for i in
               range(move_num)]
    pop += pop_tmp

    return pop

# if __name__ == '__main__':
#     best_solution = ga([10, 5])
#     print(best_solution)
