# 代码给出通过集合计算一个5层全连接的神经网络带L2正则化的损失函数的计算方法  
当神经网络很复杂的时候直接定义损失函数就会很长，这样不但可读性差而且容易出错。  
所以通过tensorflow中提供的集合(collection)来完成