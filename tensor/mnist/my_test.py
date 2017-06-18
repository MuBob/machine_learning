import tensorflow as tf

x = [17.3, 22.7, 16.6, 23.1, 24.8, 23.5, 26.4, 26.7, 20.8, 23.9,
     24.1, 23.8, 19.7, 18.5, 20.1, 21.9, 21.0, 18.6, 26.3, 25.0]

a = tf.constant(x)
mean = tf.reduce_mean(input_tensor=a)
max = tf.reduce_max(input_tensor=a, axis=0)
min = tf.reduce_min(input_tensor=a, axis=0)
sess = tf.InteractiveSession()
# 记得先调用初始化的方法，是方法。。。要加括号的！
sess.run(tf.initialize_all_variables())
num_mean=sess.run(mean)
num_max=sess.run(max)
num_min=sess.run(min)
num_ji=num_max-num_min
print("平均值：%s\n最大值：%s\n最小值：%s\n极值：%s" % (num_mean, num_max, num_min, num_ji))
sess.close()
