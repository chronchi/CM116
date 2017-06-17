"""let's implement the vgg16 cnn
using tensorflow
"""

import tensorflow as tf


"""
we'll be initializing the
weights using the a normal distribution
with 0 mean and 10^-2 variance
"""

#defining some functions before implementing the whole net

def convolvgg(name, l_input, w, b):
    cov = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)

def max_pool(name, l_input):
    return tf.nn.max_pool(l_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)


def vgg16(_X, _dropout, n_classes, imagesize, img_channel):
    #first we create the weights
    _weights = {
        'wc1': tf.Variable(tf.random_normal([3,3,img_channel,64])),
        'wc2': tf.Variable(tf.random_normal([3,3,64,64])),
         #after this one there's max pooling
        'wc3': tf.Variable(tf.random_normal([3,3,64,128])),
        'wc4': tf.Variable(tf.random_normal([3,3,128,128])),
         #after this one max pooling
        'wc5': tf.Variable(tf.random_normal([3,3,128,256])),
        'wc6': tf.Variable(tf.random_normal([3,3,256,256])),
        'wc7': tf.Variable(tf.random_normal([1,1,256,256])),
         #apply max pooling
        'wc8': tf.Variable(tf.random_normal([3,3,256,512])),
        'wc9': tf.Variable(tf.random_normal([3,3,512,512])),
        'wc10': tf.Variable(tf.random_normal([1,1,512,512])),
         #apply max pooling
        'wc11': tf.Variable(tf.random_normal([3,3,512,512])),
        'wc12': tf.Variable(tf.random_normal([3,3,512,512])),
        'wc13': tf.Variable(tf.random_normal([1,1,512,512])),
         #apply max pooling
        'wd1': tf.Variable(tf.random_normal([7*7*512,4096])),
        'wd2': tf.Variable(tf.random_normal([4096,4096])),
        'out': tf.Variable(tf.random_normal([4096,n_classes]))
    }
    _biases = {
        'bc1': tf.Variable(tf.zeros([64])),
        'bc2': tf.Variable(tf.zeros([64])),
        'bc3': tf.Variable(tf.zeros([128])),
        'bc4': tf.Variable(tf.zeros([128])),
        'bc5': tf.Variable(tf.zeros([256])),
        'bc6': tf.Variable(tf.zeros([256])),
        'bc7': tf.Variable(tf.zeros([256])),
        'bc8': tf.Variable(tf.zeros([512])),
        'bc9': tf.Variable(tf.zeros([512])),
        'bc10': tf.Variable(tf.zeros([512])),
        'bc11': tf.Variable(tf.zeros([512])),
        'bc12': tf.Variable(tf.zeros([512])),
        'bc13': tf.Variable(tf.zeros([512])),
        'bd1': tf.Variable(tf.zeros([4096])),
        'bd2': tf.Variable(tf.zeros([4096])),
        'out': tf.Variable(tf.zeros([n_classes])),
    }
    #Reshape input picture
    _X = tf.reshape(_X, shape=[-1,imagesize,imagesize,img_channel])
    # 1st Convolution layer
    conv1 = convolvgg('conv1', _X, _weights['wc1'], _biases['bc1'])
    # 2nd Convolution layer
    conv2 = convolvgg('conv2', conv1, _weights['wc2'], _biases['bc2'])
    # Now we apply the first max_pooling
    pool1 = max_pool('pool1', conv2)
    # 3rd Convolution layer
    conv3 = convolvgg('conv3', pool1, _weights['wc3'], _biases['bc3'])
    #4th Convolution layer
    conv4 = convolvgg('conv4', conv3, _weights['wc4'], _biases['bc4'])
    #2nd max pooling
    pool2 = max_pool('pool2', conv4)
    #5th Convolution layer
    conv5 = convolvgg('conv5', pool2, _weights['wc5'], _biases['bc5'])
    #6th Convolution layer
    conv6 = convolvgg('conv6', conv5, _weights['wc6'], _biases['bc6'])
    #7th Convolution layer
    conv7 = convolvgg('conv7', conv6, _weights['wc7'], _biases['bc7'])
    #3rd max pooling
    pool3 = max_pool('pool3', conv7)
    #8th Convolution layer
    conv8 = convolvgg('conv8',pool3, _weights['wc8'], _biases['bc8'])
    #9th Convolution layer
    conv9 = convolvgg('conv9',conv8, _weights['wc9'], _biases['bc9'])
    #10th Convolution layer
    conv10 = convolvgg('conv10',conv9, _weights['wc10'], _biases['bc10'])
    #4th max pooling
    pool4 = max_pool('pool4', conv10)
    #11th Convolution layer
    conv11 = convolvgg('conv11', pool4, _weights['wc11'], _biases['bc11'])
    #12th Convolution layer
    conv12 = convolvgg('conv12', conv11, _weights['wc12'], _biases['bc12'])
    #13th Convolution layer
    conv13 = convolvgg('conv13', conv12, _weights['wc13'], _biases['bc13'])
    #5th max pooling
    pool5 = max_pool('pool4', conv13)
    #reshape the pool5 to connect to FC
    pool5_flat = tf.reshape(pool5,[-1,_weights['wd1'].get_shape().as_list()[0]])
    #1st FC
    fc1 = tf.nn.relu(tf.matmul(pool5_flat, _weights['wd1'])+_biases['bd1'], name='fc1')
    #first dropout
    fc1drop = tf.nn.dropout(fc1, _dropout)
    #2nd FC
    fc2 = tf.nn.relu(tf.matmul(fc1drop,_weights['wd2'])+_biases['bd1'], name='fc2')
    #second dropout
    fc2drop = tf.nn.dropout(fc2, _dropout)
    #Output
    out = tf.matmul(fc2drop, _weights['out']) + _biases['out']
    return out
