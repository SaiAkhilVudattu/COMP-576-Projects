from scipy import misc
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import imageio
import matplotlib.pyplot as plt
import matplotlib as mp
import random


# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE

    init = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(init)
    return W


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    init = tf.constant(0.1, shape)
    b = tf.Variable(init)
    return b


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')
    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    return h_max


ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 50

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'CIFAR10-1/CIFAR10/Train/%d/Image%05d.png' % (
            iclass, isample)
        im = imageio.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot lable
    for isample in range(0, ntest):
        path = 'CIFAR10-1/CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = imageio.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

sess=tf.compat.v1.InteractiveSession()

tf_data = tf.placeholder(tf.float32, [None, imsize, imsize, nchannels])  # tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, [None, nclass])  # tf variable for labels

# --------------------------------------------------
# model
# create your model

# Convolutional layer with kernel 5 x 5 and 32 filter maps followed by ReLU
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.tanh(conv2d(tf_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer with kernel 5 x 5 and 64 filter maps followed by ReLU
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully Connected layer that has input 7*7*64 and output 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully Connected layer that has input 1024 and output 10
# Softmax layer (Softmax Regression + Softmax Nonlinearity)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(tf_labels, h_fc2)
cross_entropy = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
# optimization

sess.run(tf.initialize_all_variables())
batch_xs = np.zeros((batchsize, imsize, imsize, nchannels))  # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize, nclass))  # setup as [batchsize, the how many classes]

for i in range(1000):  # try a small iteration size once it works then continue
    perm = np.arange(ntrain * nclass)
    np.random.shuffle(perm)
    loss_list = []
    accuracy_list = []
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]
        loss = cross_entropy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        acc = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        first_weight = W_conv1.eval()
        loss_list.append(loss)
        accuracy_list.append(acc)
    if i % 10 == 0:
        # calculate train accuracy and print it
        print('step %d, loss %g, training accuracy %g' % (i, loss, acc))
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})  # dropout only during training

# --------------------------------------------------
# test

activation1 = h_conv1.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
activation2 = h_conv2.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})

print("test accuracy %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

sess.close()

figure, x1 = plt.subplots()
x1.plot(range(len(accuracy_list)), accuracy_list, 'k', label='accuracy for AdamOptimizer')
plt.show()

figure, x2 = plt.subplots()
x2.plot(range(len(loss_list)), loss_list, 'k', label='loss for AdamOptimizer')
plt.show()

# Plot the filters of the first layer
figure = plt.figure()
for i in range(32):
    x1 = figure.add_subplot(4, 8, 1 + i)
    x1.imshow(first_weight[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.show()

# Calculate the statistics of the activations in the convolutional layers on test images.
print("activation1: mean %g, variance %g" % (np.mean(np.array(activation1)), np.var(np.array(activation1))))
print("activation2: mean %g, variance %g" % (np.mean(np.array(activation2)), np.var(np.array(activation2))))