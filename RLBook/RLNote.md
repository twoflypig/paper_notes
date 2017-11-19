#### 什么是策略网络 ?

策略网络是 state->action 的概率

#### 什么是value function?

value function(Q)是 (state,action)->value(这个值可以是最后的reward)

#### 什么是actor-critic？

**我通俗解释一下actor-critic方法。**我用神经网络举例；实际上你可以用线性函数、kernel等等方法做函数近似。

**Actor**（玩家）**：**为了玩转这个游戏得到尽量高的reward，你需要实现一个函数：输入state，输出action，即上面的第2步。可以用神经网络来近似这个函数。剩下的任务就是如何训练神经网络，让它的表现更好（得更高的reward）。这个网络就被称为actor

**Critic**（评委）**：**为了训练actor，你需要知道actor的表现到底怎么样，根据表现来决定对神经网络参数的调整。这就要用到强化学习中的“Q-value”。但Q-value也是一个未知的函数，所以也可以用神经网络来近似。这个网络被称为critic。

**Actor-Critic的训练**。我先通俗解释一下。

1. Actor看到游戏目前的state，做出一个action。
2. Critic根据state和action两者，对actor刚才的表现打一个分数。
3. Actor依据critic（评委）的打分，调整自己的策略（actor神经网络参数），争取下次做得更好。
4. Critic根据系统给出的reward（相当于ground truth）和其他评委的打分（critic target）来调整自己的打分策略（critic神经网络参数）。

#### 什么是Q-Learning？

Q-learning是迭代的来近似value function(Q),是off-policy

#### 什么是on-policy,off-policy？

