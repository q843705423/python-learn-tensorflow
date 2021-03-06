#coding:utf-8
import tensorflow as tf
# 滑动平均
w1 = tf.Variable(0,dtype=tf.float32)
global_step = tf.Variable(0,trainable=False)
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
ema_op = ema.apply(tf.trainable_variables())
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print (sess.run([w1,ema.average(w1)]))
    sess.run(tf.assign(w1,1))
    sess.run(ema_op)
    print( sess.run([w1,ema.average(w1)]))
    sess.run(tf.assign(w1,10))
    print (sess.run([w1,ema.average(w1)]))

    sess.run(ema_op)
    print (sess.run([w1,ema.average(w1)]))
    sess.run(ema_op)
    print (sess.run([w1,ema.average(w1)]))
    sess.run(ema_op)
    print (sess.run([w1,ema.average(w1)]))

    sess.run(tf.assign(w1,100))
    for i in range(20):
        sess.run(ema_op)
        print (sess.run([w1,ema.average(w1)]))


