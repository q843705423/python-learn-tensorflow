import tensorflow as tf
import numpy as np
size = 8
seed = 23455
cost = 1
prpfit = 9

rmd = np.random.RandomState(seed)
X = rmd.rand(32,2)
Y = [[x1+x2+rmd.rand()/10.0-0.05] for (x1,x2) in X]

x = tf.placeholder(tf.float32,shape=(None,2))

y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

loss = tf.reduce_sum(tf.where(tf.greater(y,y_)),(y-y_)*cost,(y-y_)*prpfit)
train_step = tf.train.GradientDescentOptimizer(0.001).minmize(loss)
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for i  in range(20000):
        st = (i*8)%32
        en = (i*8)%32+8
        w1_val = sess.run(w1,feed_dict={x: X[st:en],y_: Y_[st:en]})
        print "after %d step,the w=%f "% (i,w1_val)
