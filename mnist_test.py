import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INERVAL_SECS =5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
        y = tf.mnist_forward.forward(x,None)

        ema = tf.train.ExponentiaMovingAverage(mnist_backward.MOVE_AVERAGE_DACAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict={x: mnist.test.images, y_:mnist_test.labels})
                    print("after %s training step ,test accuracy = %g"%(global_step,accuray_score))
                else:
                    print("no checkpoint found")
                    return
            time.sleep(TEST_INTERVAL_SECS)
def main():
    mnist = input_data.read_data_sets('./data',one_host=True)
    test(mnist)
if __name__ == '__main__':
    main()
                
                

        

