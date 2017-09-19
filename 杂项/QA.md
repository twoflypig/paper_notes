can embedding correctly discribe the true relation with word's neighbors?

### What is cross-entory?

下式中$x$为样本，$y$为真实标签，C表示的是在样本上二分类问题的代价函数
$$
\begin{eqnarray} C = -\frac{1}{n} \sum_x \left[y \ln a + (1-y ) \ln (1-a) \right], \tag{57}\end{eqnarray}
$$

如果目标可以同时是多个值，即**多类别多分类**问题，比如说[1,1,0,0,0],那么上式可以拓展为
$$
\begin{eqnarray} C = -\frac{1}{n} \sum_x \sum_j \left[y_j \ln a^L_j + (1-y_j) \ln (1-a^L_j) \right]. \tag{63}\end{eqnarray}
$$
用来表示多类别中多分类问题，而下式可以用来表示**多类别单分类**问题
$$
\begin{eqnarray} C = -\frac{1}{n} \sum_x \sum_j \left[y_j \ln a^L_j  \right]. \tag{64}\end{eqnarray}
$$
[neural-networks-and-deep-learning](https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s1.html)

[wiki:cross entrpopy](https://en.wikipedia.org/wiki/Cross_entropy#Motivation)

### What is Batch Normalization?

Batch Normalization 意味着，对与特定的神经元，在mini-batch中输入的样本在这个神经元的输出的分布强制转换为一个特定的分布。

### What is Matrix derivative





more details see:

[知乎](https://www.zhihu.com/question/39523290)

[Kronecker product](https://zh.wikipedia.org/wiki/%E5%85%8B%E7%BD%97%E5%86%85%E5%85%8B%E7%A7%AF)

[矩阵求导](http://xuehy.github.io/2014/04/18/2014-04-18-matrixcalc/)



### Why LSTM can vanish gradient explosion and 