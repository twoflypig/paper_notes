# Prefix

[来自博客](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)

# How wide should our Tensorflow graph be?

由链式法则可知，误差向后传播时会发生梯度爆炸，所以要向后传播有限个步骤，所以有time_step这个参数。在tensorflow中会把图的宽度限制为n个单元，即输入长度为n的输入，然后在每次迭代后做反向传播。这就是说我们把输入序列为49，把它分成7个子序列，分为7次输入计算，在每个图中来自第7步的错误会传播到全部的7步中

A natural interpretation of backpropagating errors a maximum of nn steps means that we backpropagate every possible error nn steps. That is, if we have a sequence of length 49, and choose n=7n=7, we would backpropagate 42 of the errors the full 7 steps. *This is not the approach we take in Tensorflow.* Tensorflow’s approach is to limit the graph to being nn units wide. See [Tensorflow’s writeup on Truncated Backpropagation](https://www.tensorflow.org/versions/r0.9/tutorials/recurrent/index.html#truncated-backpropagation) (“[Truncated backpropagation] is easy to implement by feeding inputs of length [nn] at a time and doing backward pass after each iteration.”). This means that we would take our sequence of length 49, break it up into 7 sub-sequences of length 7 that we feed into the graph in 7 separate computations, and that only the errors from the 7th input in each graph are backpropagated the full 7 steps. Therefore, even if you think there are no dependencies longer than 7 steps in your data, it may still be worthwhile to use n>7n>7 so as to increase the proportion of errors that are backpropagated by 7 steps. For an empirical investigation of the difference between backpropagating every error nn steps and Tensorflow-style backpropagation, see my post on [Styles of Truncated Backpropagation](https://r2rt.com/styles-of-truncated-backpropagation.html).

# Using lists of tensors to represent the width

使用列表最好

