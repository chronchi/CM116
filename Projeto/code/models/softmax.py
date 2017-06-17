#Let's do a softmax model for our data

import tensorflow as tf
import Image
import numpy as np
import numpy.random as npr
import scipy.misc as spm
import PIL

#next_batch function
def next_batch(i,path,num_channels):
    length_of_i = i.size
    xtrain = np.empty([length_of_i,224*224*num_channels],dtype=np.float32)
    ytrain = np.empty([length_of_i,2],dtype=np.float32)
    counter = np.int(0)
    for sample in i:
        img = Image.open(path + "/" + str(sample) + ".jpg")
        if img.mode == 'CMYK':
            img = img.convert('RGB')
        img = img.resize((224,224))
        img = np.array(img,dtype=np.float32)
        xtrain[counter,:] = np.concatenate(np.concatenate(img))
        if sample < 100:
            ytrain[counter,:] = np.array([1.0,0.0])
        elif 100 <= sample:
            ytrain[counter,:] = np.array([0.0,1.0])
        counter += 1
    return xtrain/255, ytrain
#initialize the shuffling for the training

size_of_vector = 272
vector = range(0,size_of_vector)
vector = npr.choice(vector, size_of_vector, replace=False)
vector_training = vector[0:220]
vector_test = vector[220:size_of_vector]

#save the vectors above

np.save("vectors/vector", vector)
np.save("vectors/vector_training", vector_training)
np.save("vectors/vector_test", vector_test)

#graph of our model

graph = tf.Graph()

input_size = 224*224*3  #tamanho do input
classes = 2 #number of classes

x_input = tf.placeholder(tf.float32, shape = [None, input_size])
y_input = tf.placeholder(tf.float32, shape = [None, classes])

W = tf.Variable(tf.zeros([input_size, classes]))
b = tf.Variable(tf.zeros([classes]))

#We will be the cross entropy as loss function

y_final = tf.matmul(x_input, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_final))



#let's use the adam optimizer to minimize our loss

lr = 1e-4 #learning rate

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

#correct predictions
correct_prediction = tf.equal(tf.argmax(y_final,1), tf.argmax(y_input,1))

#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

#now we iterate over the number of epochs with a batch size of 110

epochs = 50
batch_size = 220
path = "sample"

vector_training = np.load("vectors/vector_training.npy")

for epoch in range(epochs):
    counter = 0
    for i in range(0,vector_training.size,batch_size):
        batch = next_batch(vector_training[i:(i+batch_size)],path,3)
        if counter%2 == 0:
            train_accuracy = accuracy.eval(session=sess,feed_dict={
                x_input:batch[0], y_input: batch[1]})
            print("step %d, training accuracy %g"%(counter, train_accuracy))
            train_loss = cross_entropy.eval(session=sess,feed_dict={
                x_input:batch[0], y_input: batch[1]})
            print("step %d, training loss %g, epoch %d"%(counter, train_loss, epoch))
        train_step.run(session=sess, feed_dict={x_input: batch[0], y_input: batch[1]})
        counter += 1

#save the model
saver = tf.train.Saver()
saver.save(sess, "ZParametrosoft/my_model")

#load the model
new_saver = tf.train.import_meta_graph('ZParametrosoft/modelo_softmax.meta') # first load the graph
new_saver.restore(sess, tf.train.latest_checkpoint('./ZParametrosoft'))


vector_test = np.load("vectors/vector_test.npy")

xtest, ytest = next_batch(vector_test,path,3)
print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x_input: xtest, y_input: ytest}))

cross_entropy.eval(session=sess, feed_dict={x_input: xtest})
result = sess.run(y_final, feed_dict={x_input: xtest})
