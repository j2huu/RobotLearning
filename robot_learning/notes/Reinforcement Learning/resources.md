# RL 原始资料与阅读范围

[目录](README.md)

先读对应讲义，再按问题查原文，不要求一开始通读全部材料。以下以教材、官方课程/文档和算法原始论文为主；浏览核对日期：2026-10-25。资料能在线访问不代表 notebook 已执行。

## 主线

| 资料 | 对应内容 | 怎么用 |
|---|---|---|
| [Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed.](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) | 第 2 章 bandit；3–4 MDP/DP；5–7 MC/TD/n-step；13 PG | 查定义、假设与表格算法推导 |
| [Spinning Up：Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) | R1/R2 | state/observation、return、value |
| [Spinning Up：Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) | R5/R6 | 轨迹概率、log-derivative、baseline |
| [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) | RL、IL、offline、model-based | 按专题查对应 lecture，课程页面可能随学期更新 |
| [Gymnasium：Handling Time Limits](https://gymnasium.farama.org/v1.3.0/tutorials/gymnasium_basics/handling_time_limits/) | R3/R6 | 明确 terminal/truncation 与 bootstrap；该页为 1.3.0 文档 |

作者的 [Sutton & Barto 书页](https://incompleteideas.net/book/the-book-2nd.html) 是另一个入口，本次访问返回 502，因此不把它列作已成功读取的在线正文。教材章节编号依第二版。

## 算法论文与实现说明

| 内容 | 原始资料 | 对应课 |
|---|---|---|
| DQN | [2013 Atari 论文](https://arxiv.org/abs/1312.5602)；[2015 Nature](https://www.nature.com/articles/nature14236) | R4；两版细节不同，独立 target network 对应 2015 版本。Nature 页面本次受访问跳转限制 |
| Double DQN | [van Hasselt et al.](https://arxiv.org/abs/1509.06461) | R4 |
| Dueling / PER | [Dueling](https://arxiv.org/abs/1511.06581)、[Prioritized Replay](https://arxiv.org/abs/1511.05952) | R4 扩展 |
| GAE | [Schulman et al.](https://arxiv.org/abs/1506.02438) | R6 |
| PPO | [原文](https://arxiv.org/abs/1707.06347)、[Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | R7 |
| DDPG | [Lillicrap et al.](https://arxiv.org/abs/1509.02971) | R8 |
| TD3 | [Fujimoto et al.](https://arxiv.org/abs/1802.09477) | R8 |
| SAC | [ICML 2018 版本](https://arxiv.org/abs/1801.01290)、[Algorithms and Applications](https://arxiv.org/abs/1812.05905)、[Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) | R8；本专题公式用后续常见双 Q、无独立 V 版本 |
| Shaping | [Ng, Harada & Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | R9 |
| RND / HER | [RND](https://arxiv.org/abs/1810.12894)、[HER](https://arxiv.org/abs/1707.01495) | R9 |
| DAgger | [Ross et al.](https://proceedings.mlr.press/v15/ross11a.html) | R10 |
| Offline RL | [CQL](https://arxiv.org/abs/2006.04779)、[IQL](https://arxiv.org/abs/2110.06169) | R10 |

各课数值、环境、诊断题和代码是自行构造的教学材料，不是论文实验复现。面试优先级是学习安排建议，不代表公司题库或可验证的出现频率。

## 你的Walker项目：2026-11-09核对

[项目课](07d_dm_walker_project.md) 以本人的`hw4/walker.py`为主，既有 [NIO实现复盘](../../../../JOB/interview/NIO_interview/ppo_walker_implementation_review.md) 作为辅助；本次没有重新训练或恢复历史依赖环境。

| 原始资料 | 核对什么 |
|---|---|
| [DM Control Walker](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/walker.py) | observation字段、reward、任务与control timestep |
| [walker.xml](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/walker.xml) | body/joint/actuator顺序、control范围和physics timestep |
| [Physics.velocity](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py) | velocity返回qvel，而非只取六个腿关节速度 |
| [MuJoCo actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model) | control、传动与关节广义力的关系 |

当前main的语义用于交叉检查；它不证明项目当年的dm_control/MuJoCo版本。原实现、教学GAE参考函数、合成notebook分别标明边界。
