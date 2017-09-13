#Abstract

之先流行的方法是使用RNN去处理输入序列到一个可变长度的输出序列。我们引入了一个**完全基于CNN的结**构。与RNN相比，在所有元素上的计算在训练是可以做到并行处理，并且优化时更加简单，因为非线性单元是固定的，输入长度也是独立的。我们使用gated linear units decoder layer 去消除梯度传播并且我们对每个decoder layer都装备了一个attention。

# 1 Introduction

text summarization:Rush et al., 2015; Nallapati et al., 2016; Shen et al., 2016 

Convolutional networks do **not depend on** the computations of the previous time step and therefore allow parallelization over every element in a sequence. This contrasts with RNNs which maintain a hidden state of the entire past that prevents parallel computation within a sequence

**Hierarchical structure** provides a shorter path to capture long-range dependencies compared to the chain structure modeled by recurrent networks. e.g. we can obtain a feature representation capturing relationships within a window of n words by applying only O( nk ) convolutional operations for kernels of width k, compared to a linear number O(n) for recurrent neural networks.Fixing the number of nonlinearities applied to the inputs also eases learning.

Some people partially used convolutional and still use rnn as their decoder . And perform are great.[Gehring et al., 2016]

In this paper we propose an architecture for sequence to sequence modeling that is entirely convolutional. Our model is equipped with gated linear units (Dauphin et al., 2016) and residual connections (He et al., 2015a). We also use attention in every decoder layer and demonstrate that each attention layer only adds a negligible amount of overhead. The combination of these choices enables us to tackle large scale problems (§3)

*Dataset*:

- WMT’16 English-Romanian translation 
- WMT’14 English-German  
- WMT’14 English-French 

better and faster

# 2 Rnn 

pass

# 3 A Convolutional Architecture

 totally based on cnn

## 3.1 Position Embeddings

First, we embed input elements x = (x1, . . . , xm) in distributional space as w = (w1, . . . , wm), where wj ∈ Rf is a column in an embedding matrix D ∈ RV ×f 

Seconed.We also equip our model with a sense of order by embedding the absolute position of input elements p = (p1, . . . , pm) where pj ∈ Rf (**use position embedding**)

## 3.2 Convolutional Block Structure

We denote the output of the lth block as hl = (hl 1, . . . , hl n) for the decoder network, and zl = (z1l , . . . , zml ) for the encoder network

we refer to blocks and layers interchangeably



