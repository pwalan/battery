import tensorflow as tf
import pandas as pd

BATCH_SIZE = 10
TEST_SIZE = 100


def get_XY():
    x = []
    y = []
    input_path = "/Users/alanp/Projects/高比功率电池项目/数据/Train/10001737_output.csv"
    data_frame = pd.read_csv(input_path)
    data_values = data_frame.values
    for i in range(data_values.shape[0]):
        data_raw = data_values[i]
        tmp_x = []
        tmp_y = []
        # for j in range(3, 13):
        #     tmp_x.append(data_raw[j])
        tmp_x.append(data_raw[3] / 60000)
        tmp_x.append(data_raw[4] / 100)
        tmp_x.append(data_raw[5] / 100)
        tmp_x.append(data_raw[7] / 50)
        tmp_x.append(data_raw[8] / 50)
        tmp_x.append(data_raw[9] / 400)
        tmp_x.append(data_raw[10] / 300)
        tmp_x.append(data_raw[11] / 605)
        tmp_x.append(data_raw[12] / 605)
        tmp_y.append(data_raw[14])
        x.append(tmp_x)
        y.append(tmp_y)
    return x[:len(x) - TEST_SIZE], y[:len(x) - TEST_SIZE]


def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans


# 获取输入和真实值
X, Y_ = get_XY()

input_num = 9  # 输入参数个数
hid_num1 = 7  # 隐藏层1节点数
hid_num2 = 3  # 隐藏层2节点数
output_num = 1  # 输出个数

# 定义神经网络的输入、参数和输出,定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, input_num))
y_ = tf.placeholder(tf.float32, shape=(None, output_num))

# w1 = tf.Variable(tf.random_normal([input_num, hid_num], stddev=1.0, seed=1))
# w2 = tf.Variable(tf.random_normal([hid_num, output_num], stddev=1.0, seed=1))
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)

# h1 = addLayer(x, input_num, hid_num1, activity_function=tf.nn.relu)
h1 = addLayer(x, input_num, hid_num1)
# h2 = addLayer(h1, hid_num1, hid_num2)
y = addLayer(h1, hid_num1, output_num)

# 定义损失函数及反向传播方法。
# 均方误差损失函数
loss_mse = tf.reduce_mean(tf.square(y - y_))
# 优化算法训练参数
global_step = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(1.0, global_step, 500, 0.96, staircase=True)
learning_rate = 0.1
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_mse)
train_step = tf.train.FtrlOptimizer(learning_rate).minimize(loss_mse)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_mse, global_step=STEPS)

# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 训练模型
    STEPS = 10000
    print("开始训练")
    for i in range(STEPS + 1):
        start = (i * BATCH_SIZE) % len(Y_)
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            # 每训练500个steps打印训练误差
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on train data is %g" % (i, total_loss))
            # print(sess.run(w1), sess.run(w2))

    # 进行预测
    print("进行预测")
    test_x, test_y = get_XY()
    test_x = test_x[len(test_x) - TEST_SIZE:]
    test_y = test_y[len(test_y) - TEST_SIZE:]
    total_loss = sess.run(loss_mse, feed_dict={x: test_x, y_: test_y})
    print("loss_mse on test data is " + str(total_loss))
    for i in range(TEST_SIZE):
        print(test_y[i], sess.run(y, feed_dict={x: [test_x[i]]}))
