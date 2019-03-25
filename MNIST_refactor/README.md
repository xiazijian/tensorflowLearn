结合变量管理机制和tensorflow模型持久化机制，实现一个tensorflow训练神经网络模型的最佳实践。  
防止在训练过程程序死机，在训练过程中每隔一段时间保存一次模型训练的中间结果。  
将训练和测试分成2个独立程序。  
重构后的MNIST分成3个程序：  
mnist_inference.py  定义前向传播的过程以及神经网络的参数  
mnist_train.py  定义神经网络训练过程  
mnist_eval.py  定义测试过程  
