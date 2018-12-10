import tensorflow as tf
import forward as fw
import pandas as pd

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 9))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    # X, Y_=
    df = pd.read_csv("sample_data.csv")
    frames = [df["MAXVO2_"], df["CHILDREN"], df["HEIGHT3"], df["_AGE80"], df["NOBCUSE6"], df["_FRUTSU1"], df["_BMI5"],
              df["WEIGHT2"], df["FRUTDA2_"]]
    X = pd.concat(frames, axis=1)
    Y_ = df["_RFHYPE5"]

    y = fw.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)

    # 采用指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, BATCH_SIZE, LEARNING_RATE_DECAY,
                                               staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(ce, global_step=global_step)


    # 训练
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 200
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(cem, feed_dict={x: X, y_: Y_})
                print("after %d steps:,loss is :%f", i, loss_v)

if __name__ == '__main__':
    backward()




