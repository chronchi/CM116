"""
using vgg16.py to predict curitiba's sightseeing
"""

import tensorflow as tf
import numpy as np
from next_batch5 import *
from vgg16 import *

learn_rate = 0.01
decay_rate = 0.1
batch_size = 43
display_step = 10

n_classes = 5
dropout = 0.5   #as in the article
imagesize = 224 #as in the article
img_channel = 3 #as in the article

#let's create the placeholders for the inputs

x = tf.placeholder(tf.float32, [None, imagesize*imagesize*img_channel])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32) #placeholder for the dropout

pred = vgg16(x, keep_prob, n_classes, imagesize, img_channel)

#we will use cross entropy without l2 regularization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

#now we set the adam optimizer with learning rate decay
global_step = tf.Variable(0,trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 250, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)


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
epochs = 10
path = '../dataset/sample'


vector_training = np.load('../dataset/vectors/vector_training5.npy')
vector_test = np.load('../dataset/vectors/vector_test5.npy')
xtest, ytest = next_batch(vector_test,path,3)

for epoch in range(epochs):
    for i in range(0,vector_training.size,batch_size):
        batch_xs, batch_ys = next_batch(vector_training[i:(i+batch_size)],path,3)
        if step % 44 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            #rate = sess.run(lr)
            print("lr " + str(learn_rate) + " Iter " + str(step) + ", Minibatch Loss= "
            + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            + ", Epoch " + str(epoch))
            print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
                x: xtest, y: ytest, keep_prob: 1.}))
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        step += 1

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
