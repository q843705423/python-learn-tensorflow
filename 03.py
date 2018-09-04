import tensorflow as tf
x = tf.placeholder(tf.float32,shape=(None,2))
w1 = tf.Variable(tf.random_normal([2,3],stddev=2,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=2,seed=1))

a = tf.matmul(x,w1)
a = tf.matmul(a,w2)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print("ans:\n",sess.run(a,feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.4,0.5]]}))
