# Abstract

RNN为什么难以优化。本文提出了一个技术，利用dropout来解决这个问题。即使用两个相同的RNN(共享同样的参数)，然后用不同的dropout masks(这是啥)，最小化他们之间预测(pre-softmax)的差距。如何保持对不同dropout mask的健壮性？结果显示这个正则项的上界就是dropout目标的线性期望，这样就解决了训练和推断中dropout的不同导致的问题？？

# Introduce

batch normalization 效果并不像在FFN那里那么成功，尽管是有一点效果的。同样的，naive dropout在RNN中也是ineffective。

1，在多层RNN中对于非循环的连接应用dropout。2，在训练中队整个句子使用同样的dropout mask？？3，在权重矩阵上使用dropout操作。4，和dropout的精神相同，但是使用上一个神经元的输出而不是当前的。5，作为batch normalization的替代，layer normalization balabala. RNN的batch normalization  balala 。 activity and temporal activation regularization 是有效的方法。

本文是最小化两个相同参数但不是同dropout mask的LSTM的预测损失的带权和？？，然后L2正则加在两个网络的预测上的差别上。分析表明这个正则相当于最小化来自两个相同网络的不同dropout mask的预测方差。因此就是鼓励网络invariant to dropout masks。

# Fraternal Dropout

dropout 在densely connected layers更加有效是因为CNN会更容易拟合。但是RNN中dropout在训练和推断时会有不同，因为推断的时候 assumes linear activations to correct for the factor by which the expected value of each activation would be different Ma et al. (2016) 不懂。通常情况，模型的预测能力会随着不同的dropout mask而不同。然而，理想情况下是预测与dropout mask无关。

在fraternal dropout背后的思想就是，训练一个神经网络模型，使得它们之间的在不同dropout mask情况下的预测的方差尽可能的小。

# 3 Related Work

## 3.1 RELATION TO EXPECTATION LINEAR DROPOUT (ELD)

这里的意思就是说，有人在2016年提出了一个使用单个drop out 来逼近 平均的 drop out，缺点就是一个网络里面输入了两次。

而本文提出的方法的上界就是它的4倍

## 3.2 RELATION TO Π-MODEL

这个模型的本意是提高在分类任务上的表现的。模型就是想利用未标记的数据来最小化两个不同drop out mask网络的预测差。与本文的模型的差别就是，它没有把另一个网络的损失项作为loss

# 4 Experiments

在两个数据集上 Penn Tree-bank (PTB)和 WikiText-2(WT2)  用的是AWD-LSTM 3-layer



# 5 Ablation Studies

single layer LSTM





