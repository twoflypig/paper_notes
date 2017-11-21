# Abstract

RNN为什么难以优化。本文提出了一个技术，利用dropout来解决这个问题。即使用两个相同的RNN(共享同样的参数)，然后用不同的dropout masks(这是啥)，最小化他们之间预测的差距。如何保持对不同dropout mask的健壮性？结果显示这个正则项的上界就是dropout目标的线性期望，这样就解决了训练和推断中dropout的不同导致的问题？？

# Introduce

batch normalization 效果并不像在FFN那里那么成功，尽管是有一点效果的。同样的，naive dropout在RNN中也是ineffective。

1，在多层RNN中对于非循环的连接应用dropout。2，在训练中队整个句子使用同样的dropout mask？？3，在权重矩阵上使用dropout操作。4，和dropout的精神相同，但是使用上一个神经元的输出而不是当前的。5，作为batch normalization的替代，layer normalization balabala. RNN的batch normalization  balala 。 activity and temporal activation regularization 是有效的方法。

本文是最小化两个相同参数但不是同dropout mask的LSTM的预测损失的带权和？？，然后L2正则加在两个网络的预测上的差别上。分析表明这个正则相当于最小化来自两个相同网络的不同dropout mask的预测方差。因此就是鼓励网络invariant to dropout masks。

# Fraternal Dropout

dropout 在densely connected layers更加有效是因为CNN会更容易你和。

