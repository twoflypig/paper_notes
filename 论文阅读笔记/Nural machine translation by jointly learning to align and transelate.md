# Abstract

在传统的RNN编码器与解码器翻译任务中，我们认为中间生成的固定长度的向量C是一个效果的瓶颈，所以我们提出了能使模型自动的从源句子中找到与预测部分相关的部分，在没有明确的把这些相关部分进行分割。

# 1 Introduction

为了解决这个固定长度向量的问题，这个新模型:每次模型输出一个词，会在源句子中找到相关信息的位置。模型然后会基于这些源位置的上下文向量和所有之前产生的目标词预测一个新的词。

这个模型新的区别在于，它把输入句子编码到一个向量序列并且在解码时找到适应的子序列

# 2 Background:Neural Machine Translation

条件概率问题

介绍了下RNN

# 3 Learning to align and translate

把双向RNN作为编码器，还有在一个在翻译时会在源句子进行搜索的解码器

## 3.1 Decoder:General Description

上下文向量依赖于编码器把输入句子映射到的sequence of annotation($h_1$,...,$h_{Tx}$)，每个标注$h_i$包含整个输入序列的信息，并带有对输入句子的第i个词附近的强烈关注。内容向量是这个annotation $h_i$的带权之和、![tu](../photo/attention2014/1.png)

每个权值$a_{ij}$ 由下式计算

![tu](../photo/attention2014/2.png)

其中

![tu](../photo/attention2014/3.png)

是一个 alignment model，能够对位置j的输入和输出位置i的匹配进行打分。分数基于RNN的隐藏层状态$s_{i-1}$和输入序列的第j个标注$h_j$。

我们把alignment model a 参数化为一个神经网络，这个网络和其他的部件一起训练。注意这个alignment model不是一个latent variable，相反的alignment model 直接计算一个soft alignment,允许进行梯度计算

我们知道这个带权和的annotation实际就是一个期望的annotation，是最有可能的alignments(对齐)。认为$a_{ij}$是目标词$y_i$的概率，或者翻译自源词$x_j$。那么第i个context vector $c_i$就是所有annotations上带有概率$a_{ij}$的期望annotation

(We can understand the approach of taking a weighted sum of all the annotations as computing an
expected annotation, where the expectation is over possible alignments. Let αij be a probability that
the target word yi is aligned to, or translated from, a source word xj. Then, the i-th context vector
ci is the expected annotation over all the annotations with probabilities αij. )

就好像一个注意力机制，减轻了编码器的压力(???)

## 3.2 Encoder:bidirectional rnn for annotating sequences

双向RNN包含前向和反向两个RNN，前向网络按输入序列顺序读入网络，并且计算前向的hidden states，反向网路RNN从输入序列的倒序读入，会产生一个反向的hidden states。这样，标注$h_j$包含了前面的词和后面的词两种信息。(图中中括号的两个分别代表前向和后向隐藏层输出)

![tu](../photo/attention2014/4.png)

根据RNNs更好的表达最近输入的趋势，标注$h_j$将会更加关注$x_j$周围的词。这个标注序列会被后面的解码器和对齐模型用来训练context vetor (5,6)



