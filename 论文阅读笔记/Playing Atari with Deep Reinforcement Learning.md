# Pre

修改时间:2017年7月14日19:45:35

版本v1.2

# Abstract

第一个从深度模型，使用卷积神经网络，把图像的原始像素作为输入，用神经网络作为Q值函数的估计

# Introduction

深度学习的突破，利用了很多结构例如convolutional networks, multilayer perceptrons, restricted Boltzmann machines and recurrent neural networks 。几个比较多大的挑战。第一:大多数成功的深度神经网络要求大量标记的训练数据。但是增强学习必须从数值型的reward上(经常是稀疏的，带有噪声和延迟的)。在行为和最终的结果之间，可能会有几千个timestep长的延迟，和在监督学习中的输入和目标之间的直接关联相比，这看起来就特别的令人畏惧。另一个问题就是增强学习遇到的是**高度相关** 的序列(而深度学习假设数据样本之间是无关的)。而且，在RL中随着算法的改变数据的分布也会发生改变(而深度学习假设的是一个固定的分布)

本文证明了卷积神经网络能够克服这些挑战从原始的录像数据中学习到成功的控制策略。训练算法是Q-learning算法的辩题，用SGD去更新参数。针对数据的关联和非固定分布问题，我们使用经验回放(随机从之前的转换中采样)，因此平滑了过去行为的训练分布。

# Background

注意通常上游戏的分数依赖于之前整个的动作和观察序列;关于一个行为的反馈可能会在几千步以后才会得到。

单独靠屏幕的输入是很难理解当前的状态的，因此我们还考虑了行为和观察构成的序列，假定是有限的。那么就可以用MDP的方法去做。

机器人的目标就是选择行为去最大化未来的回报。我们假设未来的回报会因为每一步而得到衰减$\gamma$ ，并且定义了在t时间时未来打过折扣的回报为$R_T$ = $\sum_{t^{‘}=t}^{T} \gamma^{t^{'}-t} r_t^{'} \qquad $,其中T是游戏终止的时间。我们定义最佳的行为-值函数 $Q^{*} (s,a)$ 作为执行某个策略后的最大的期望返回,after一些序列以后然后执行动作a,

 $Q^{*} (s,a)$  = $max_\pi E[R_t | s_t = s ,a_t =a,\pi ]$ ,其中$\pi$ 是一个把序列映射到行为的策略，

最优的action-value同时还有一个重要的身份 *Bellamn equation* 。这是基于这样的直觉: 如果在下一个time-step $s^{’}$ 的最优 $Q^{*} (s^{'},a^{'})$  是对所有的actions $a^{‘}$ 是可知的，那么最优的策略就是去选择行为 $a^{'}$ 来最大化 r+$\gamma Q^{*}(s^{'},a^{’})$  的期望值

 $Q^{*} (s,a)$  = $E_{s^{'} - \varepsilon}  [R_t | s_t = s ,a_t =a,\pi ]$   

很对增强学习算法背后的思想是去估计action-value的函数，通过使用Bellman equation作为迭代更新, $Q_{i+1} (s,a)$  = $E[r+ \gamma Q^{*}(s^{'},a^{’}]$ . 这样 值迭代算法 会收敛到最优的值函数中，在i$\to \propto $ 时 $Q_I \to Q^{*} $ 。在实际中，这个基本的方法是不适用的，因为action-value函数是对于每个序列分别的估计，没有任何的通用性。相反的，大家常使用函数近似来估计action-value函数,$Q(s,a;\theta) = Q^{*} (s,a)$ . 社区中经常使用线性函数近似，有时也用非线性函数近似，例如神经网络。我们使用一个带有权值参数$\theta$ 的神经网络作为近似函数，as Q-network .  Q-network能够在每次迭代的情况下最小化序列的损失 $L_i (\theta_i)$ 来训练

$ L_i (\theta_i)  = E_{s,a-p(\cdot)}[ ( y_i -Q(s,a;\theta_i)  )^2]$  

其中$y_i$ = $E_{s^{‘} - \varepsilon}[r+ \gamma max_{a^{’}}Q^{*} (s^{'},a^{’},\theta_{i-1})|s,a]$ 是迭代i的目标并且$\rho(s,a)$ 对序列s和行为a的可能分布(我们成为行为分布).来自之前$\theta_{i-1}$的 参数是固定的当我们在最小化损失函数 $L_i(\theta_i)$ 。注意目标依赖神经网络的权值；这和监督学习中目标的用法相反(在学习开始前是固定的)。通过对权值进行偏微分我们可以得到下式

(3)

不是对所有的样本进行迭代，我们使用SGD

# Related Work

TD-gammon不具有广泛性

# Deep Reinforcement Learning

first,经验中的每步可能会在很多次的权值更新中被使用，这就允许更佳的数据效率。second,直接从连续的样本中进行学习是无效的，因为在样本之间有很强的关联性;随机化采样能够打破关联性。third，当我们在线学习时当前的参数会决定下一个参数训练基于的数据样本。例如，如果最大化行为是向左移动，那么训练样本大部分就会变成左边的;最大化行为向右也是同理。很容易能发现不想要的反馈循环会发生并且参数可能会卡在一个局部的最小值，或者灾难性的发散。通过使用经验回放行为的分布会在之前的状态上平均化，平滑学习过程并且避免参数的发散或者震荡。注意当我们根据经验回放进行学习时，很有必要去进行离线学习(因为我们当前的参数和产生样本的参数是不同的)，这样就可以驱动Q-learning

在实际中，我们的算法仅保存最近的N个经验元组，在执行更新的时候随机从D中进行采样。这个方法在某些方面是有限的因为内存并不能区分去重要的转换并且总是会覆盖最近的转换因为有限的内存N。一个更加复杂的采样策略可能是强调我们可以学习到最大的转换，就与prioritized sweeping 一样

## preprocessing and model architecture

对输入进行降维，转成灰度图像，图像裁剪

讲了下用神经网络来近似Q函数的几个结构

在几个游戏中结构不变，但是对reward进行了修改，因为每个游戏的score不同，正的rewards归一化到1，负的rewards归一化到-1。以这种方式处理reward限制了错误的derivatives并且使得同样的学习速率在大量的游戏上能够改动少。同时，它也会影响我们agent的表现因为它不能够区分不同数量级的reward

贪婪率在刚开始的1million frames上是1到0.1，然后在此后固定在0.1。我们训练了总共10million frames并且使用1million最近 frames的经验回放

我们使用跳帧技术，就是每隔k帧机器人会看见并且选择一个行为而不是每一帧都这么做(系统的频率类似)。因为运行一次模拟器比选择一个动作要省时间，这个技术就允许agent能够在大概k倍的游戏而不用显著的增加运行时间。
# Experiments
## Training and Stability

在强化学习中，精确的评估一个agent在训练中的过程很具有挑战性。因为我们的评估标准，我们在训练过程中定期的计算它。整个reward的平均值可能是充满噪声的因为策略的权值的微小改动能够引起策略探索状态的巨大改变。
## visualizing the value Function
pass
## Main Evaluation

我们把我们的结果和RL算法中的最好的模型进行了比较。

在固定贪婪率为0.05的步骤时的平均分数作为比较(意思就是说贪婪率设置为0.05，即最好的状态进行最终结果的对比)。

# My Test Project

注意迭代更新的在代码上是这样的

在target net 中,也就是训练的动作的目标Q值。这个目标值得定义就是当前的reward+下一个状态s_对应的最大Q值(为什么是这个呢，因为在下一个s_状态时，如果我们知道了所有要采取的action的对应的Q值，那么最优的方法就是选择一个action区最大化预计值，所以选择了Qmax) * 一个衰减系数

```python
q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  
```

在估计出来的网络中

```
a_one_hot = tf.one_hot(self.a, depth=self.n_actions, dtype=tf.float32)
self.q_eval_wrt_a = tf.reduce_sum(self.q_eval * a_one_hot, axis=1)  # shape=(None, )
```

定义的loss

```python
self.loss=tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_wrt_a,name='TD_error'))
```

