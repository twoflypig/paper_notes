# Prefix

来自博文: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

# Some base knowledge about RNN

$x_t$ 表示第几步的(step)的输入。如果有28个timestep，那么unfold后就会有28个节点。本节点还有一个$o_t$ 的输出，是隐藏层的输出，最后输出层的输出才可以作为两层LSTM的输入x(博客中是这样理解的),可以认为$S_t$是网络的记忆单元。

如何训练RNNs,同普通的训练过程相同，只不过隐藏层的计算多加了一点。



训练过程:[中文翻译](http://nooverfit.com/wp/%E6%AF%8F%E4%B8%AA%E4%BA%BA%E9%83%BD%E8%83%BD%E5%BE%92%E6%89%8B%E5%86%99%E4%B8%80%E4%B8%AAlstm-rnn%E9%80%92%E5%BD%92%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-%E6%89%8B%E6%8A%8A%E6%89%8B%E6%95%99%E4%BD%A0/) 此链接翻译自  [原文出处](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)



# Recurrent Neural Networks

讲了一下直觉，人们对于文章的理解不是从零开始的，而是依赖于之前的信息。我们的想法具有persistence

传统的神经网络不能够做到依赖之前的信息，这看起来就是这个主要的缺点。

RNN能够解决这个问题，它们是带有循环的网络，使得信息能够保持(persist) ![图1](RNN-rolled.png)

在上述的图表中，是神经网络的chunk,对A输入x则会输出一个h。循环允许信息从网络中的一步传递到下一步。

一个RNN可以被认为是同一网络复制了多遍，每个网络都会传递信息给下一个网络，如下图

![图2](RNN-unrolled.png)

链状的属性就非常符合序列和列表这样的结构。(This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists. They’re the natural architecture of neural network to use for such data.)

RNN实现了很多成就，可以看这个http://karpathy.github.io/2015/05/21/rnn-effectiveness/

# The problem of Long-Term Dependencies

RNN的魅力之一就是它可能能够把之前的信息与现在信息相关联，但是这样可以嘛？

有时，我们只需要去根据最近的信息去执行当前的任何。例如，考虑一个语言模型想要去根据之前的词预测下一个词。如果我们想要去预测最后一个词在"the clound in the sky".我们不需要任何更进一步的信息，很明显最后一个词就是sky。在这样的例子中，需要的相关信息和位置的需求很小，RNNs能够去使用过去的信息。

![图3](RNN-shorttermdepdencies.png)

但是考虑一个我们需要更多上下文信息的场景。讲了一个预测语言种类的例子。在相关信息和位置(where it is needed to be )之间的gap可能会很多(意思就是说如果一个词的依赖变得和上文很多地方有关的话，就不太好确定该依赖哪个信息)

不幸的是，随着gap变大，RNNs变得不能够学习到具体的信息

理论上，RNNs绝对能够处理这样 long-term dependencies。通过仔细的调整参数就能处理这种问题的简单情况。然而，实际上RNNs看起来不能够学习到这些。

# LSTM Networks

Long short term networks LSTM  能够处理这种长期依赖的问题。

LSTM是明确的设计来避免long-term dependency的问题。记住长时间的信息是LSTMs的默认行为，而不是偶尔它们努力去学的。

所有的RNN都会链式的重复模块的结构，在标准的RNNs中，这些重复的模块将会有一个非常简单的结构，例如单个tanh层。

![图三](LSTM3-SimpleRNN.png)

LSTM也拥有链式结构，并且重复的模块有一个不同的结构。LSTMs的模块拥有4个层，以一种非常特别的方式相互影响。

![图5](LSTM3-chain.png)

下面是一些需要理解的标记

![图6](LSTM2-notation.png)

# The Core Idea Behind LSTMs

LSTM的关键是cell state，下图中水平线

cell state就像是传输带。它贯穿整个链,且会有一些微小的线性操作。信息能够很容易的不变的穿过。

![图7](LSTM3-C-line.png)

LSTM并不能移除或者添加信息给cell state，而是有叫做门的结构仔细的规定。

Gates是一种随意的让信息流动的方式。它们由一个sigmoid 的神经网络层和一个pointwise的乘法操作构成

![图](LSTM3-gate.png)

sigmoid层输出0-1之间的数，用来描述每个元件应该通过多少。0意味的全部禁止通过，1意味着全部通过

# Step-by-Step LSTM Walk Through

LSTM的第一步是用 forget gate layer来确定我们要从cell state中丢弃什么信息。它的输入时$h_{t-1}$ 和$x_t$ ,对cell state $C_{t-1}$ 中的每个数会输出一个0-1之间的数。1代表保持，0代表遗忘

在一个语言预测模型中，cell-state可能会包含当前主语的类别，以便正确的代词能够被使用。当我们看见一个新的主语时，我们想要去忘记旧主语的类别.

![图](LSTM3-focus-f.png)

下一步就是确定我们要在cell state中保存什么新信息。这可分为两个部分。首先，一个叫做输入门层的sigmoid层决定更新哪些值，然后，一个tanh层会创建一个新的候选的能够加入state的向量$\tilde{C_t}$ 。在下一步中，我们会结合这两个part去更新state。

例如在语言模型中，我们想要去增加新的主语的类别来代替我们正要忘记的旧主语。

![图](LSTM3-focus-i.png)



现在我们要更新旧的cell state $C_{t-1}$到 $C_t$ 。

把旧的cell state乘以$f_t$ ,意思是先忘记一些我们想要早点忘记的东西。然后我们加上 $i_t$ * $\tilde{C_t}$ ，这是新的候选值，通过我们是想决定更新每个状态值的多少来确定系数(scaled by how much we decided to update each state value)

在语言模型中，这就是我们实际上丢弃旧主语的gender的信息并且增加新信息

![图](LSTM3-focus-C.png)

最后，我们要决定我们输出什么。输出将基于我们的cell state，但是会是一个经过处理的版本。首先，运行 sigmoid layer来决定我们想要输出的cell state的部分。然后，我们把cell state 通过tanh(将值归一化到-1与1之间)然后把它乘以sigmoid gate的输出，以便输出我们想要输出的部分。

![图](LSTM3-focus-o.png)

# Variants on Long Short Term Memory

pass