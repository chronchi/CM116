"""
now we should try with 4 classes and softmax
"""

import tensorflow as tf
import numpy as np
from next_batch5 import *

input_size = 224*224*3  #tamanho do input
classes = 5 #number of classes

# Placholders for the data
x = tf.placeholder(tf.float32, name="x")
y_ = tf.placeholder(tf.float32, name="y_")

#Initialize weight and bias variables
W = tf.Variable(tf.zeros([input_size,classes]), name="W")
b = tf.Variable(tf.zeros([classes]), name="b")

#And the output will be
y_hat = tf.add(tf.matmul(x, W),b, name="y_hat")

#Writing down the cross entropy and giving a name for it

cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=y_hat), name="cross_entropy")

#and finally the learning rate and solver

lr = tf.placeholder(tf.float32, name="lr")

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(cross_entropy)

tf.add_to_collection('train_step', train_step)

#for counting the correct predictions
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1), name="correct_prediction")

#and for the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

#defining our parameters

size_of_vector = 597

vector = range(0,size_of_vector)
vector = npr.choice(vector, size_of_vector, replace=False)
vector_training = vector[0:477]
vector_test = vector[477:size_of_vector]

np.save('vectors/vector_training5', vector_training)
np.save('vectors/vector_test5', vector_test)
np.save('vectors/vector5', vector)

#vector_training = np.load('vectors/vector_training5.npy')
#vector_test = np.load('vectors/vector_test5.npy')

epochs = 50 #100 são muitas épocas, overfitting
batch_size = 159
path = "sample"
learn_rate = 1e-4


#initializing the variables
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

xtest, ytest = next_batch(vector_test,path,3)

for epoch in range(epochs):
    counter = 0
    for i in range(0,vector_training.size,batch_size):
        batch = next_batch(vector_training[i:(i+batch_size)],path,3)
        if epoch%2 == 0:
            if i%2 == 0:
                train_accuracy = accuracy.eval(session=sess,feed_dict={
                    x:batch[0], y_: batch[1]})
                train_loss = cross_entropy.eval(session=sess,feed_dict={
                    x:batch[0], y_: batch[1]})
                print("step %d, training loss %g, training accuracy %g, epoch %d"%(counter, train_loss, train_accuracy, epoch))
                print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
                    x: xtest, y_: ytest}))
        train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})
        counter += 1

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: xtest, y_: ytest}))

"""
the test accuracy obtained is 0.63333,
that means, 63.3%
"""

#for the confusion matrix
test = np.zeros(ytest.shape[0], dtype = np.int)
c = sess.run(tf.nn.softmax(sess.run(y_hat, feed_dict={x: xtest})))
predicted = np.zeros(ytest.shape[0], dtype=np.int)

for i in range(ytest.shape[0]):
    predicted[i] = np.argmax(c[i])
    test[i] = np.argmax(ytest[i])

a = tf.confusion_matrix(test,predicted)
sess.run(a)

"""the confusion matrix is
[ 6, 10,  1,  3,  1]
[ 1, 31,  5,  2,  0]
[ 1,  2, 10,  0,  1]
[ 2,  4,  0, 22,  0]
[ 1, 10,  0,  0,  6]
"""

saver = tf.train.Saver()
saver.save(sess, "savedmodels5class/softmax/my_model")
