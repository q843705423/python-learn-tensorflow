import tensorflow as tf
x = tf.placeholder(tf.float32,shape=[1,2])
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print("ans=",sess.run(y,feed_dict={x:[[0.7,0.5]]}))
