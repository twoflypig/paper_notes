## memory network



## APPLYING DEEP LEARNING TO ANSWER SELECTION: A STUDY AND AN OPEN TASK

### abstract

此论文在不依赖任何语言工具的情况下，应用深度学习的方法来解决非事实问答任务，并且发布一个在保险领域的数据集。在测试集上数据最好能达到65.3%。

### 1 introduction

本文从文本搭配和选择的角度来解决QA问题，本文利用深度学习来完成对回答的选择。问题的定义:给定一个问问题q和回答候选{a1,a2,a3,.....as}，目标是在候选中找到最好的回答，如果选中的回答是真实答案中的子集，那么就认为问题被正确的回答了。从定义中，QA也可以被认为是一个二分类问题。

发布的数据集包含24981个不重复的问题

### 2 model description

接下来会给出一个深度学习框架，主要思想就是，学习一个给定问题和回答候选的分布向量表达，并且使用一个相似度量度来测量匹配程度。

#### 2.1 baseline system

- bag of words : 先用 "Distributed representations of words and phrases and their compositionality" 中提到的方法训练word embedding ，然后为问题和问题的候选答案的词向量产生idf权重之和, 这样就把问题和答案都向量化了。最后就是计算每个问题和候选对之间的cosine相似度，最高的就是答案。
- information retrieval :使用了目前最好的带权依赖模型 WD, WD模型使用基于术语和术语近似的排名特征来给每个候选回答打分。基本的思想是，在问题中重要的二元组或者三元组

## Teaching Machine to Read and Comprehend

有个问题，它是怎么训练的？是要最终的嵌入向量和答案响亮进可以相同？

### Abstract

2015年还缺乏大规模数据，本文给出一个大规模的数据集并且提出了一个新的方法来解决大数据集的瓶颈

### 1 Introduction

由于缺乏大数据，人工合成的数据很难转到实际对话中，因为难以capture自然语言中的复杂性以及噪声。

本文利用文章中的总结和文句，以及相关的文章，就构成了一个三元组。实验结果显示我们神经网络的方法要好的多。

### 2 Supervised training data for reading comprehension

#### 2.1 Entity replacement and permutation

在回答一些问题时，可以不看文章而直接利用自己的已有的知识就进行回答，例如 鱼的油能够帮助抵抗X？汤精能够帮助打败X吗？ngram 的语言模型能够在无视文件内容的情况下直接回答出 cancer因为这在数据集中是一个非常频繁的词。所以，我们对语料做如下处理

- 使用一个融合系统来建立各自的共同点数据点
- 根据共同出现的情况用抽象实体标记替代所有实体
- 每当加载数据点时，随机排列这些实体标记。

### 3 Models

两个baseline.一个是选择在文件中出现频率最高的实体,然而独占多数只会选择在文件中出现频率最高而不是查询中。

#### 3.1 Symbolic Matching Models

以下几个是传统的NLP方法模型

##### Frame-Semantic Parsing : 不懂

##### Word Distance Benchmark : 测量问题与和问题对齐的实体周围上下文的距离。这个方法可以一试

#### 3.2 Neural Network Models

提出了三个模型。把NLP中的问题看成是分类问题

##### The Deep LSTM Reader

![1](photo/QA_photo/17-10-9-1.png)

先把文件按一个一个词输入到LSTM encoder中，在一个分割符后然后输入查询到encoder中。结果就是这个模型把文件和查询作为一个长句来处理。

使用了skip-connection,把每个输入x(t)连接到每一个隐藏层。

> 看上图，其实非常简单，就是用一个两层LSTM来encode query|||document或者document|||query，然后用得到的表示做分类。

##### The Attentive Reader

![1](photo/QA_photo/17-10-9-2.png)

结合最近的ATTENTION 把隐藏层固定宽度的向量变成一个带权向量,然后文章一个向量，问题一个向量。

> 这个模型将document和query分开表示，其中query部分就是用了一个双向LSTM来encode，然后将两个方向上的last  hidden  state拼接作为query的表示，document这部分也是用一个双向的LSTM来encode，每个token的表示是用两个方向上的hidden state拼接而成，document的表示则是用 document中所有token的加权平均来表示，这里的权重就是attention，权重越大表示回答query时对应的token的越重要。然后用document和query的表示做分类。

##### The impatient Reader

![1](photo/QA_photo/17-10-9-3.png)

> 这个模型在Attentive Reader模型的基础上更细了一步，即每个query token都与document 
> tokens有关联，而不是像之前的模型将整个query考虑为整体。感觉这个过程就好像是你读query中的每个token都需要找到document中对应相关的token。这个模型更加复杂一些，但效果不见得不好，从我们做阅读理解的实际体验来说，你不可能读问题中的每一个词之后，就去读一遍原文，这样效率太低了，而且原文很长的话，记忆的效果就不会很好了。