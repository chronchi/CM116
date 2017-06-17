"""Let's define a simple convolutional
neural network and see how it predicts
the images
"""

import tensorflow as tf
import Image
import numpy as np
import numpy.random as npr
import scipy.misc as spm
from next_batch import *

"""functions for initializing the
variables and for the convolutional/
max pooling ops
"""

def weight_variable(shape,stringg):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=stringg)

def bias_variable(shape, stringg):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=stringg)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #it keeps the format of the image

def max_pool_2x2(x, stringg):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=stringg) #it keeps the format of the image


#defining the graph

x = tf.placeholder(tf.float32, name="x")
y_ = tf.placeholder(tf.float32, name="y_")

#reshape the input data
x_image = tf.reshape(x, [-1,224,224,3], name="x_image")

#create variables
W_conv1 = weight_variable([5,5,3,16], "W_conv1")
b_conv1 = bias_variable([16], "b_conv1")

#apply a convolution and after that the relu
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1, name="h_conv1")

#now the max pooling op
h_pool1 = max_pool_2x2(h_conv1,"h_pool1")

#and now we want to create a fully connected layer
h_pool1_size = np.array(h_pool1.get_shape().as_list()[1:4])
reduced = reduce(lambda x, y: x*y, h_pool1_size)

Wfc_1 = weight_variable([reduced, 1024], "Wfc_1")
bfc_1 = bias_variable([1024], "bfc_1")

#flattening the pooled image
h_pool1_flat = tf.reshape(h_pool1, [-1,reduced])

#and now we can multiply the weight and sum the bias
hfc_1 = tf.add(tf.matmul(h_pool1_flat, Wfc_1), bfc_1)
hfc_1 = tf.nn.relu(hfc_1, name="hfc_1")

Wfc_2 = weight_variable([1024,2], "Wfc_2")
bfc_2 = bias_variable([2], "bfc_2")

y_hat = tf.add(tf.matmul(hfc_1,Wfc_2),bfc_2,name="y_hat")

#let's define the loss function, optimizer and accuracy

cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=y_hat), name="cross_entropy")

lr = tf.placeholder(tf.float32, name="lr")

optimizer = tf.train.AdamOptimizer(1e-6)
train_step = optimizer.minimize(cross_entropy)
tf.add_to_collection('train_step', train_step)

#for counting the correct predictions
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1), name="correct_prediction")

#and for the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

#defining our parameters
epochs = 100
batch_size = 220
path = "sample"

#define the saver
saver = tf.train.Saver()

#initializing the variables
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

#load the number of images which will be
#used for training
vector_training = np.load("vectors/vector_training.npy")

#Let's train!

for epoch in range(epochs):
    counter = 0
    for i in range(0,vector_training.size,batch_size):
        batch = next_batch(vector_training[i:(i+batch_size)],path,3)
        if counter%2 == 0:
            train_accuracy = accuracy.eval(session=sess,feed_dict={
                x:batch[0], y_: batch[1]})
            print("step %d, training accuracy %g"%(counter, train_accuracy))
            train_loss = cross_entropy.eval(session=sess,feed_dict={
                x:batch[0], y_: batch[1]})
            print("step %d, training loss %g, epoch %d"%(counter, train_loss, epoch))
        train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})
        counter += 1
    #save after every epoch
    saver.save(sess, "cnnsimple/my_model")

print("Optimization finished")

vector_test = np.load("vectors/vector_test.npy")

xtest, ytest = next_batch(vector_test,path,3)
print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: xtest, y_: ytest}))

#test accuracy 0.538462

result = sess.run(y_hat, feed_dict={x: xtest})
