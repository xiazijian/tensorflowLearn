代码中有一行计算交叉熵  
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)  
这里面logits参数使用的是没有经过滑动平均的神经网络前向传播的结果  
这里不理解为啥不用经过滑动平均的神经网络前向传播的结果average_y  
但是换成average_y后准确率低到与随机差不多  
