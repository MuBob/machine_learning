from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)
# 一、实现回归模型
# 通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个：
# 用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784])
print("x=",x)
# 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改
# 权重值
W = tf.Variable(tf.zeros([784,10]))
print("权重值W=",W)
# 偏置量
b = tf.Variable(tf.zeros([10]))
print("偏置量b=",b)
# 赋予tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置
y = tf.nn.softmax(tf.matmul(x,W) + b)
print("得到的初始化学习值y=",y)
# 首先，我们用tf.matmul(​​X，W)表示x乘以W，对应之前等式里面的，这里x是一个2维张量拥有多个输入。然后再加上b，把和输入到tf.nn.softmax函数里面。

# 二、训练模型
print("二、开始训练")
y_ = tf.placeholder("float", [None,10])
print("训练得到y=",y_)
# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
print("通过学习值获取学习前后的交叉熵cross_entropy=", cross_entropy)
# TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
print("梯度下降算法得到的下降梯度step=", train_step)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
# 让模型循环训练1000次
print("准备循环训练1000次")
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print("当前第%d次运行"%(i))
# 三、评估模型
# 首先让我们找出那些预测正确的标签
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
print("评价预测结果", correct_prediction)
# 为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("预测结果=",accuracy)
# 最后，我们计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))