# Let's try the save and restore models of tensorflow with optimizers
# We will do a small classification problem using cross entropy
# and Adam Optimizer

#inspired on http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

import tensorflow as tf
import numpy as np
from next_batch import *

input_size = 224*224*3  #tamanho do input
classes = 2 #number of classes

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

epochs = 1
batch_size = 220
path = "sample"
learn_rate = 1e-4


#initializing the variables
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)


vector_training = np.load("vectors/vector_training.npy")

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

#Let's save the model now
saver = tf.train.Saver()
saver.save(sess, "testesave/my_model")

#Now we want to reload the variables and ops
sess = tf.Session()

new_saver = tf.train.import_meta_graph('testesave/my_model.meta') # first load the graph
new_saver.restore(sess, tf.train.latest_checkpoint('./testesave'))

sess.run(tf.global_variables_initializer())

#And we can get the variables and operations by using their names
print(sess.run('b:0')) #for example

#And we can build again our graph getting the placeholders,
#variables and ops using get_tensor_by_name(name of the tensor),
#but first we have to define

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")

W = graph.get_tensor_by_name("W:0")
b = graph.get_tensor_by_name("b:0")

cross_entropy = graph.get_tensor_by_name("cross_entropy:0")

#defining the train step again
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

train_step = tf.get_collection("train_step")[0]

correct_prediction = graph.get_tensor_by_name("correct_prediction:0")
accuracy = graph.get_tensor_by_name("accuracy:0")

#defining our parameters

epochs = 100
batch_size = 220
path = "sample"
learn_rate = 1e-4


#now we can keep training

vector_training = np.load("vectors/vector_training.npy")

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

vector_test = np.load("vectors/vector_test.npy")

xtest, ytest = next_batch(vector_test,path,3)
print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: xtest, y_: ytest}))

#accuracy 0.75

#and so we can save it again, overwriting in this case
saver = tf.train.Saver()
saver.save(sess, "testesave/my_model")
