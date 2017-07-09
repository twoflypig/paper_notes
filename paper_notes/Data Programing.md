以labeling function的形式去编辑 weak supervision.即把不同的弱学习方法看做是labeling functions.但是这样就会出现conflict 和在error rates上很大范围上不同。为了解决这个问题，通过学习labeling functions之间关联的结构，就可以达到自动去除噪声的目的。

​	动机的原因之一就是，用户对选择特征很苦恼。

# 2 Related Work

Distant supervision:在文本中的关系抽取中，KB中的已知关系的基础就是启发式去映射到输入语料中

一个问题：怎样在不知正确标记下去对不同的experts的准确度进行建模

co-training: 通过选择两个条件无关的数据的角度，来利用一小部分标记数据和一大堆无标记数据

Boosting : 和本文中的方法很像，但是本文不需要已经标记的数据。

本文考虑了更加通用的场景，大量的noisy标记函数会有冲突和依赖

# 3 Data Programming 范例

两个大挑战：

- 标记的数据很少
- 相关的外部知识库不足
- application specifications 很flux,需要我们去变化模型

在这样的前提下，我们想要一个好方法。

## Example 3.1

举了一个简单的文本关系抽取的例子。任务是识别gen和疾病mention的co-occurring是一个随机的关系或者不是。a1是Distant Learing (高准确度),a2是一个单纯的分层的方法去标记更多的例子(低精度)，最后a3是一个混合的函数，利用a1和a2

labeling function并不需要特别好，形式可以多种多样，但是会有矛盾和冲突

## 独立的labeling function

对于给定的真实的标记类，我们首先描述了标记函数独立的标记。在这个模型下，每个标记函数$$ a_i$$ 会有概率去标记一个目标和概率去正确的标记这个目标

## Noise-Aware Enpirical Loss

已知我们的学习参数已经成功的找到了一些$$\hat{a}$$  和 $$\hat{\beta}$$ 能够精准的描述训练集，现在我们就能开始去估计参数 w ,在最小化线性模型的期望风险。因此定义了对噪声敏感的empirical risk. 公式看的不是太明白，好像是从概率的角度上说明的

# Handling Dependencies

当用户添加更多的labeling function的时候，这些函数之间会存在依赖，通过对这种依赖建图，可以使得系统产生更好的参数估计

## Label Function Dependency Graph

## Mideling Dependencies 

依赖关系的出现使得我们不再能够使用简单的贝叶斯网络对我们的labels建模。相反的，我们把分布建模为factor graph。

## Learing with Dependencies

# Experiments

证明三个声明:

- 能够建立高质量的机器学习系统
- 能够与自动生成特征的方法相结合
- intuitive and productive framework





