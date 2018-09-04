import tensorflow as tf

LERAN_RATE_BASE = 0.1
LERAN_RATE_DECAP = 0.99
LERAN_RATE_STEP = 1

global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(LERAN_RATE_BASE,global_step,LERAN_RATE_STEP,LERAN_RATE_DECAP,staircase=True)

W = tf.Variable(tf.constant(5,dtype=tf.float32))
loss = tf.square(W+1)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for i in range(4000):

        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(W)
        loss_val = sess.run(loss)
        print "after %s steps,global_step=%f,w=%f,learning_rate=%f,loss=%f" % (i,global_step_val,w_val,learning_rate_val,loss_val)
        
