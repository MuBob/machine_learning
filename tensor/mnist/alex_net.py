# Copy by link http://blog.csdn.net/crazyice521/article/details/52870658
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)

import tensorflow as tf

#define the parameter of the NET
learning_rate = 0.001
training_iters = 200000   #iters
batch_size = 64  # the number of the train_data in Every iterarion
display_step = 20

n_input = 784 #the dimentional of the input
n_classes =10 # the class of the label
dropout = 0.8

x = tf.placeholder("float", [None,n_input])  #this is easy to change the train data
y = tf.placeholder("float", [None,n_classes])
keep_prob = tf.placeholder("float")

#define the conv operate
def conv2d(name,I_inpuit, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(I_inpuit,w,strides=[1,1,1,1],padding='SAME'),b),name=name)

def max_poll(name, I_input,k):
    return tf.nn.max_pool(I_input,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)

def norm(name,I_input,lsize=4):
    return tf.nn.lrn(I_input,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75,name=name)

#define the framework of the network
def Alex_net(X, weights,biases,dropout):
    X = tf.reshape(X,shape=[-1,28,28,1])

    #layer 1
    conv1 = conv2d('conv1',X ,weights['wc1'],biases['bc1'])
    pool1 = max_poll('pool1',conv1, k=2)
    norm1 = norm('norm1',pool1,lsize =4)
    norm1 = tf.nn.dropout(norm1,dropout)

    #layer 2
    conv2 = conv2d('conv2',norm1,weights['wc2'],biases['bc2'])
    pool2 = max_poll('pool2',conv2,k=2)
    norm2 = norm('norm2',pool2,lsize=4)
    norm2 = tf.nn.dropout(norm2,dropout)

    #layer 3
    conv3 = conv2d('conv3',norm2,weights['wc3'],biases['bc3'])
    pool3 = max_poll('pool3',conv3,k=2)
    norm3 = norm('norm3',pool3,lsize=4)
    norm3 = tf.nn.dropout(norm3,dropout)

    #full-connected network
    fc1 = tf.reshape(norm3, [-1, weights['wf1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['wf1'])+biases['bf1'],name='fc1')
    fc2 = tf.nn.relu(tf.matmul(fc1,weights['wf2'])+biases['bf2'],name='fc2')

    #output layer

    out = tf.matmul(fc2,weights['out'])+biases['out']

    return out
    #save the parameters
weights = {
    'wc1':tf.Variable(tf.random_normal([3,3,1,64])),
    'wc2':tf.Variable(tf.random_normal([3,3,64,128])),
    'wc3':tf.Variable(tf.random_normal([3,3,128,256])),
    'wf1':tf.Variable(tf.random_normal([4*4*256,1024])),
    'wf2':tf.Variable(tf.random_normal([1024,1024])),
    'out':tf.Variable(tf.random_normal([1024,10]))
    }

biases = {
    'bc1':tf.Variable(tf.random_normal([64])),
    'bc2':tf.Variable(tf.random_normal([128])),
    'bc3':tf.Variable(tf.random_normal([256])),
    'bf1':tf.Variable(tf.random_normal([1024])),
    'bf2':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([10]))
    }
# model the learning network
pred = Alex_net(x,weights,biases,keep_prob)

#define the loss and train_step
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred, logits=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

#test the network
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuary = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#initialize the variables

init = tf.initialize_all_variables()

# TRAIN
with tf.Session() as sess:
    sess.run(init)
    step = 1
    #Keep training until reach the max iteration
    while step * batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:dropout})
        if step%display_step==0:
            acc = sess.run(accuary,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
            losses = sess.run(loss,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
            print("Iter"+str(step*batch_size)+",Minibatch_loss = "+"{:.6f}".format(losses)+",Training Accuary"+"{:.5f}".format(acc))
        step +=1
    print("Optimization Finished!")

    # Test the accuary
    print("Testing Accuary:", sess.run(accuary,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1}))
