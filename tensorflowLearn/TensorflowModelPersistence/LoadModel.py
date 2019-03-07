import tensorflow as tf
# 直接家在持久化的图
saver = tf.train.import_meta_graph('./model/firstmodel.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess,'./model/firstmodel.ckpt')
    # 通过张量的名字来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("mul:0")))