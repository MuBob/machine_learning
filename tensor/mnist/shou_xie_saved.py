from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# http://blog.csdn.net/ying86615791/article/details/72731372
meta_path = './model/checkpoint/model.ckpt.meta'
model_path = './model/checkpoint/model.ckpt'
saver = tf.train.import_meta_graph(meta_path)  # 导入图

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saver.restore(sess, model_path)  # 导入变量值
    graph = tf.get_default_graph()
    prob_op = graph.get_operation_by_name('prob')  # 这个只是获取了operation， 至于有什么用还不知道
prediction = graph.get_tensor_by_name('prob:0')  # 获取之前prob那个操作的输出，即prediction
print(sess.run(prediction, feed_dict={...}))  # 要想获取这个值，需要输入之前的placeholder
print(sess.run(
    graph.get_tensor_by_name('logits_classifier/weights:0')))  # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重