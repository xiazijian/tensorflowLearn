import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
v2 = tf.Variable(tf.constant(2.,shape=[1],name='v2'))

result = v1 * v2   # print(result) 的结果：Tensor("mul:0", shape=(1,), dtype=float32)

init_op = tf.global_variables_initializer()
# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 保存模型
    saver.save(sess,'./model/firstmodel.ckpt')
    print(result)