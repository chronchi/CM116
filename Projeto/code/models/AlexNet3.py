import tensorflow as tf
import numpy as np
from next_batch3 import *

# Create AlexNet model
def conv1st(name, l_input, w, b):
    cov = tf.nn.conv2d(l_input, w, strides=[1, 4, 4, 1], padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)

def conv2d(name, l_input, w, b):
    cov = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)

def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _dropout, n_classes, imagesize, img_channel):
    # Store layers weight & bias
    _weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, img_channel, 64])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        #'wd1': tf.Variable(tf.random_normal([8*8*256, 1024])),
        'wd1': tf.Variable(tf.random_normal([5*5*256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, n_classes]))
    }
    _biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([192])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([256])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, imagesize, imagesize, img_channel])
    # Convolution Layer
    conv1 = conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=3, s=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)
    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=3, s=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)
    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv5, k=3, s=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)
    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

learn_rate = 0.001
decay_rate = 0.1
batch_size = 60
display_step = 20

n_classes = 3
dropout = 0.6 # Dropout, probability to keep units
imagesize = 224
img_channel = 3

x = tf.placeholder(tf.float32, [None, imagesize*imagesize*img_channel])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

pred = alex_net(x, keep_prob, n_classes, imagesize, img_channel)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 1000, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

sess = tf.Session()
sess.run(init)

step = 0
epochs = 200
path = 'sample'

#let's create the vector for selecting the batches

#size_of_vector = 365
#vector = range(0,size_of_vector)
#vector = npr.choice(vector, size_of_vector, replace=False)
#vector_training = vector[0:300]
#vector_test = vector[300:size_of_vector]

#np.save('vectors/vector_training3', vector_training)
#np.save('vectors/vector_test3', vector_test)
#np.save('vectors/vector3', vector)

vector_training = np.load('vectors/vector_training3.npy')
vector_test = np.load('vectors/vector_test3')


vectortrain = npr.choice(vector_training, vector_training.size, replace=False)
error = np.empty(0)

for epoch in range(epochs):
    for i in range(0,vectortrain.size,batch_size):
        batch_xs, batch_ys = next_batch(vectortrain[i:(i+batch_size)],path,3)
        if step % 1 == 0:
            #acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            #rate = sess.run(lr)
            #print("lr " + str(rate) + " Iter " + str(step) + ", Minibatch Loss= "
            #+ "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            #+ ", Epoch " + str(epoch))
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        step += 1
        error = np.append(error, loss)
        print(step)
        print("loss is "+str(loss)+" at epoch "+str(epoch))
    #if step % 100 == 0:
    #    saver.save(sess, 'savedmodels3class/alexnet/my_model', global_step=step*batch_size)

xtest, ytest = next_batch(vector_test,path,3)
print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: xtest, y: ytest, keep_prob: 1.}))

#for the confusion matrix
test = np.zeros(ytest.shape[0], dtype = np.int)
c = sess.run(tf.nn.softmax(sess.run(pred, feed_dict={x: xtest, keep_prob: 1.0})))
predicted = np.zeros(ytest.shape[0], dtype=np.int)

for i in range(ytest.shape[0]):
    predicted[i] = np.argmax(c[i])
    test[i] = np.argmax(ytest[i])

a = tf.confusion_matrix(test,predicted)
sess.run(a)

""" And the confusion matrix for running the
algorithm during 100 epochs and lr = 0.0001 is
[2, 12, 0]
[4, 30, 0]
[7, 10, 0]
"""

"""
Training with lr=0.001 and changing the order of batches
every epoch, we got a accuracy of 80%
on the test set.

the confusion matrix is given below
[ 9,  9,  0]
[ 0, 31,  0]
[ 2,  2, 12]
"""
