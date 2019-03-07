import tensorflow as tf

v2 = tf.get_variable("v",[1])
print(v2.name)
with tf.variable_scope("foo"):
    v = tf.get_variable(name="v",shape=[1])


with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable("v",[1])
    print(v1.name)

with tf.variable_scope("",reuse=True):
    v4 = tf.get_variable("foo/v",[1])
    print(v4==v1)