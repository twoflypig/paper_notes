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