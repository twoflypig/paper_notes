# Why DQN

若使用表格來存储每个状态，和在当前state下每个action所拥有的Q值，那么对于复杂问题下，状态太多。故我们可以使用神经网络来代替该表。将状态和动作当做神经网络的输入，然后经过神经网络的分析得到该动作的Q值。另一种做法是只输入状态值, 输出所有的动作值, 然后按照 Q learning 的原则, 直接选择拥有最大值的动作当做下一步要做的动作。在本次中使用第二种

# Q值更新

这一部分仅仅根据输入为状态，输出为动作的神经网络。

maze中，在选择了当前动作a后，会把当前状态s更新为新的状态s_，在s状态时，q的预测值就是q_table中(s,a)的值。我们之前已经采取了行为a,所以会得到回报r和下一个状态s\_  。那么我们在s状态时q_table训练的目标就是s\_ 状态的q_table。其中s\_ 的Q值拿s\_ 中所有action对应的Q最大的那个来近似。这样已经知道了q\_predict和q_target(估计)，就可以对q表进行更新

```python
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:
            A = choose_action(S, q_table)   # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.ix[S, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                # 拿动作发生之后的状态对应的最大的Q值作为实际的Q
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新
            step_counter += 1
            Record.append(q_table.iloc[:,1].mean())
    return q_table
```

# 两大利器

- Experience replay.将过去的经历保存下来，当做训练库。
- Fixed Q-targets.这样就会存在两个神经网络