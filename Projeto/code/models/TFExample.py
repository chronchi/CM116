"""let's modify the example given on the tensorflow website
to our problem
"""

import tensorflow as tf
import Image
import numpy as np
import numpy.random as npr
import scipy.misc as spm
import PIL

"""
add functions to be using later on
"""

#def img_reader(path):
#    return spm.imread(path)

def next_batch(i,path,num_channels):
    length_of_i = i.size
    xtrain = np.empty([length_of_i,224*224*num_channels],dtype=np.float32)
    ytrain = np.empty([length_of_i,2],dtype=np.float32)
    counter = np.int(0)
    for sample in i:
        img = Image.open(path + "/" + str(sample) + ".jpg")
        img = img.resize((224,224), PIL.Image.ANTIALIAS)
        img = np.array(img,dtype=np.float32)
        img = np.concatenate(np.concatenate(img))
        xtrain[counter,:] = img/255
        if sample < 100:
            ytrain[counter,:] = np.array([1.0,0.0])
        elif 100 <= sample:
            ytrain[counter,:] = np.array([0.0,1.0])
        #elif 9999 < sample <= 14999:
            #ytrain[counter,:] = np.array([0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
        #elif 14999 < sample <= 19999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0])
        #elif 19999 < sample <= 24999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0])
        #elif 24999 < sample <= 29999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0])
        #elif 29999 < sample <= 34999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
        #elif 34999 < sample <= 39999:
            #ytrain[counter,:] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
        counter += 1
    return xtrain, ytrain

"""initialize the shuffling for the training
"""

#random_vector = np.array(range(0,40000))

size_of_vector = 272


vector = range(0,size_of_vector)

vector = npr.choice(vector, size_of_vector, replace=False)

vector_training = vector[0:220]
vector_test = vector[220:size_of_vector]

#random_vector = npr.choice(random_vector[1:100], size = 800, replace=False)

#random_training = np.array(random_vector[0:32000])
#random_test = np.array(random_vector[32000:40000])


"""now let's use tensorflow
"""

graph = tf.Graph()

def weight_variable(shape,stringg):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=stringg)

def bias_variable(shape, stringg):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=stringg)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#with graph.as_default():
x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
W_conv1 = weight_variable([5, 5, 3, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")
x_image = tf.reshape(x, [-1,224,224,3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
"""let's add one convolution more to downsize the input
and change the filter size
"""
W_conv3 = weight_variable([5, 5, 64, 16], "W_conv3")
b_conv3 = bias_variable([16], "b_conv3")
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
keep_prob = tf.placeholder(tf.float32)
h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob)
#adding one more convolutional layer
W_conv4 = weight_variable([3,3,16,32], "W_conv4")
b_conv4 = bias_variable([32], "b_conv4")
h_conv4 = tf.nn.relu(conv2d(h_pool3_drop, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)
h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob)
"""Now we have the flc layers"""
W_fc1 = weight_variable([14 * 14 * 32, 2048], "W_fc1")
b_fc1 = bias_variable([2048], "b_fc1")
h_pool4_flat = tf.reshape(h_pool4_drop, [-1, 14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([2048, 2], "W_fc2")
b_fc2 = bias_variable([2], "b_fc2")
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#cross entropy to minimize
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#we use adam optimizer to the minimization
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.global_variables_initializer()

num_epoch = 1
path = "sample"
batch_size = 110


sess = tf.Session()
sess.run(init_op)

#saver = tf.train.Saver()

vector_training = np.array(vector_training)
sizevtrain = vector_training.size

for epoch in range(num_epoch):
    counter = 0
    for i in range(0,vector_training.size,batch_size):
        batch = next_batch(vector_training[i:(i+batch_size)],path,3)
        if counter%2 == 0:
            train_accuracy = accuracy.eval(session=sess,feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(counter, train_accuracy))
            train_loss = cross_entropy.eval(session=sess,feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training loss %g, epoch %d"%(counter, train_loss, epoch))
        train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        counter += 1

xtest, ytest = next_batch(np.array(vector_test),path,3)
print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: xtest, y_: ytest, keep_prob: 1.0}))
