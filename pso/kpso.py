# coding:utf-8

import random
import math
import copy
import time
import numpy

from pso.particle import Particle
from pso.kmeans import kmeans


def obj_func(x, data_4at_SOC):
    if len(data_4at_SOC['voltage']) < 5:
        return 10000
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
        return obj * (-1.0)


def kpso(input):
    """
    算法入口
    :param input:{'soc': cur_soc, 't': [], 'voltage': [], 'current': []}
    :return:
    """
    start_time = time.clock()
    # 算法参数
    N = 200  # 粒子个数
    iter_num = 1000  # 迭代次数
    upper_limit = [600, 0.1, 500, 0.1]  # 参数搜索上界
    lower_limit = [400, 0, 0, 0]  # 参数搜索下届
    particles = []  # 粒子信息，存储在类Particle中
    c1 = 2
    c2 = 2  # c1和c2为学习因子
    gbest = [-10000000000.0, []]  # 所有粒子中最好的值和位置
    w = 0.5  # 惯性权重，调节对解空间的搜索能力
    aggregation_threshold = 100  # 聚集度的阈值
    first_Ath = False  # 是否第一次到达聚集度阈值
    start_kmeans = False  # 开始按照分群进行操作
    K = 3  # 聚类数
    gbest_k1 = [-10000000000.0, []]  # 种群1(最近)中最好的值和位置
    gbest_k2 = [-10000000000.0, []]  # 种群2(中等)中最好的值和位置
    gbest_k3 = [-10000000000.0, []]  # 种群3(最远)中最好的值和位置

    # 初始化
    vmax = [(upper_limit[i] - lower_limit[i]) / 500.0 for i in range(len(upper_limit))]  # 最大速度
    for i in range(N):
        particle = Particle([], [], 0.0, [], 0.0)
        p = []
        v = []
        for k in range(len(upper_limit)):
            # 更新位置的每个分量
            pk = random.random() * upper_limit[k]
            if pk >= upper_limit[k]:
                pk = upper_limit[k] - vmax[k] / 100
            elif pk <= lower_limit[k]:
                pk = lower_limit[k] + vmax[k] / 100
            p.append(pk)
            # 更新速度的每个分量
            vk = random.random() * vmax[k]
            if vk >= vmax[k]:
                vk = vmax[k] - 0.1
            v.append(vk)
        particle.position = p
        particle.velocity = v
        particle.value = obj_func(particle.position, input)
        particle.best_value = particle.value
        particle.best_postion = particle.position
        # 查找所有粒子中最大值
        if gbest[0] < particle.best_value:
            gbest = [particle.best_value, particle.position]
        particles.append(particle)

    print("初始化后最优值：", gbest)

    # 将计算过程存入文件
    now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    filename = '/Users/alanp/Downloads/param/' + now + ".csv"
    f = open(filename, 'a')
    f.write("迭代次数,适应度值,计算耗时,聚集度\n")

    # 算法开始
    for i in range(iter_num):
        for_start = time.clock()
        # 上一次最优点位置
        old_best = gbest[1][:]
        aggregation = 500
        for j in range(N):
            p = particles[j].position
            v = particles[j].velocity
            pbest = particles[j].best_postion
            newV = []
            newP = []
            for k in range(len(p)):
                # 更新速度的每个分量
                if not start_kmeans:
                    vk = w * v[k] + c1 * random.random() * (pbest[k] - p[k]) + c2 * random.random() * (
                            gbest[1][k] - p[k])
                else:
                    # 在聚集度小于阈值时切换更新策略
                    vk = w * v[k] + c1 * random.random() * (pbest[k] - p[k]) + c2 * random.random() * (
                            0.5 * (gbest_k1[1][k] - p[k]) + 0.3 * (gbest_k2[1][k] - p[k]) + 0.2 * (
                            gbest_k3[1][k] - p[k]))
                if math.fabs(vk) >= vmax[k]:
                    if vk < 0:
                        vk = -vmax[k]
                    else:
                        vk = vmax[k]
                newV.append(vk)
                # 更新位置的每个分量
                pk = p[k] + vk
                if pk >= upper_limit[k]:
                    pk = upper_limit[k] - vmax[k] / 100
                elif pk <= lower_limit[k]:
                    pk = lower_limit[k] + vmax[k] / 100
                newP.append(pk)
            particles[j].velocity = newV
            particles[j].position = newP
            # 计算目标函数值
            particles[j].value = obj_func(particles[j].position, input)
            # 更新个体极值
            if particles[j].best_value < particles[j].value:
                particles[j].best_value = particles[j].value
                particles[j].best_postion = particles[j].position
            # 查找所有粒子中最大值
            if gbest[0] < particles[j].best_value:
                gbest = [particles[j].best_value, particles[j].position]
            # 计算聚集度
            v1 = numpy.array(particles[j].position)
            v2 = numpy.array(old_best)
            aggregation += numpy.sqrt(numpy.sum(numpy.square(v1 - v2)))

        # 判断聚集度是否小于阈值
        if aggregation / N < aggregation_threshold:
            if not first_Ath:
                first_Ath = True
                start_kmeans = True
            if start_kmeans:
                start_kmeans = False
                # 使用kemans进行一次分群
                print("开始分群")
                dataSet = []
                for k in range(N):
                    dataSet.append(particles[k].position)
                centroids, clusterAssment = kmeans(numpy.mat(dataSet), K)
                clusterAssment = clusterAssment.tolist()
                k_mark = []
                obj_tmps = []
                for l in range(K):
                    obj_tmps.append(abs(obj_func(centroids[l], input)))
                for l in range(K):
                    if obj_tmps[l] == min(obj_tmps):
                        k_mark.append(1)
                    elif obj_tmps[l] == max(obj_tmps):
                        k_mark.append(3)
                    else:
                        k_mark.append(2)
                print(obj_tmps)
                print(k_mark)
            # 根据分群结果计算gbest_k1、k2、k3
            for m in range(N):
                tmp = int(clusterAssment[m][0])
                if tmp == 0:
                    if gbest_k1[0] < particles[j].best_value:
                        gbest_k1 = [particles[m].best_value, particles[m].position]
                elif tmp == 1:
                    if gbest_k2[0] < particles[j].best_value:
                        gbest_k2 = [particles[m].best_value, particles[m].position]
                elif tmp == 2:
                    if gbest_k3[0] < particles[j].best_value:
                        gbest_k3 = [particles[m].best_value, particles[m].position]
            print(gbest_k1)
            print(gbest_k2)
            print(gbest_k3)
            print("...")

        print("迭代次数:" + str(i + 1))
        print("最佳值：" + str(gbest[0]))
        print("最佳点：", gbest[1])
        print("聚集度", aggregation / N)
        f.write(
            str(i + 1) + "," + str(-gbest[0]) + "," + str(time.clock() - for_start) + "," + str(aggregation / N) + "\n")

    f.close()
    time_consume = time.clock() - start_time
    print('计算耗时：', time_consume, "s")
    return gbest[1], time_consume
