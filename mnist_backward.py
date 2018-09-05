import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MODEL_SAVE_PATH = "/model/"
MODEL_NAME = 'mnist_model'

def backward(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_connection('losses'))
    learn_rate = tf.train.exponential_decay(LEARNING_RATE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staitcese=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=gloabl_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.initialize_all_vairbales()
        sess.run(init_op)
        
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = .sess.run([train_op,loss,globel_step],feed_dict = (x:xs,y_:ys))
            if i%1000==0:
                print("after %d steps,loss on training batch is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
                
