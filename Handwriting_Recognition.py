import tensorflow as tf
import numpy as np
from Dataset_functions import *
from Tensormodel import *
#Around 99% Accuracy
images, labels = list(read_mnist(dataset = 'training', path = "C:\\Users\\Jake\\Documents\\Python_data\\MNIST"))


sess = tf.InteractiveSession()
data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='data')
lbl = tf.placeholder(tf.float32, shape=[None, 10], name='lbl')
y_normalized = tf.placeholder(tf.float32, shape=[None, 10], name='y_normalized')
Training_status = tf.placeholder(tf.bool)

W_conv1 = tf.Variable(tf.truncated_normal([6,6,1,32],mean=0.0,stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.005, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(data, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# [14,14,32]

h_flat = tf.reshape(h_pool1, [-1,14*14*32])

#Fully-connected Layer
W_fc = tf.Variable(tf.truncated_normal([14*14*32, 1024],mean=0.0,stddev=0.0289))
b_fc = tf.Variable(tf.constant(0.005, shape=[1024]))
h_fc = tf.nn.relu(tf.matmul(h_flat, W_fc) + b_fc)
#shape: [images, 1024]

#Apply Drop-out
def train_keep_func(): return tf.constant(0.5, tf.float32)
def run_keep_func(): return tf.constant(1.0, tf.float32)
keep_prob = tf.cond(Training_status, train_keep_func, run_keep_func)
h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

#Read-out Layer
W_read = tf.Variable(tf.truncated_normal([1024,10],mean=0.0,stddev=0.0442))
b_read = tf.Variable(tf.constant(0.005, shape=[10]))
y_conv = tf.matmul(h_fc_drop, W_read) + b_read
#shape: [images, 10]

#Optimize, calculate accuracy and get the class prediction percentages as a set of normalized vectors
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, lbl))
Optimize = tf.train.AdamOptimizer(2e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(lbl,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_normalized = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "C:\\Users\\Jake\\Documents\\Python_Models\\MNIST\\classifier.ckpt")
print("Model restored")

#TRAINING#################################################################################################################
for i in range(10):
    image_batch, hot_label_batch = extract_random_image_batch(images, labels, [64,28,28,1], one_hot=True, num_classes=10)
    gradients, acc = sess.run([Optimize, accuracy], feed_dict={data: image_batch, lbl: hot_label_batch, Training_status: True})
    print("Batch: " + str(i) + "  Accuracy: " + str(acc))
     
save_path = saver.save(sess, "C:\\Users\\Jake\\Documents\\Python_Models\\MNIST\\classifier.ckpt")
print("Model saved in file: %s" % save_path)
#Check###############################################################3
image_batch, hot_label_batch = extract_random_image_batch(images, labels, [2,28,28,1], one_hot=True, num_classes=10)
y_norm, Weight_1 = sess.run([y_normalized, W_conv1 ], feed_dict={data: image_batch, Training_status: False})
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

print(y_norm)
plot_mnist(image_batch[0])
plot_mnist(image_batch[1])
