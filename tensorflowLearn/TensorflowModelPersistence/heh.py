import tensorflow as tf
a=tf.constant([1.0,2.0],name="a")
b=tf.constant([3.0,4.0],name="b")
c=a+b
print(a.graph)
print(c.graph)
print(tf.get_default_graph())