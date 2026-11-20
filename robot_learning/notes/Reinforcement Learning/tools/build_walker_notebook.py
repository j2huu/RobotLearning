"""Generate RL02 JSON using stdlib only; never execute notebook experiments."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
NAME = "RL02_walker_ppo_tensor_trace.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n",
            "execution_count": None, "outputs": []}


def build():
    return [
        md("""
        # RL02：你的 Walker 项目——24维输入、6维动作与一次PPO更新

        配套课程R7D。上传此单个文件到Colab，从头顺序运行；默认CPU，不挂载Drive，不读取本地仓库。
        使用合成字段与未训练的24→256→256→6/1网络，检查Gaussian、clip、GAE和梯度数据流。
        **不创建dm_control/MuJoCo环境，不训练Walker步态，不复现900+回报。**
        对齐的是项目网络尺寸和关键运算；额外assert和诊断属于教学补充。
        输出留空，首次运行后才有结果。若Colab缺包，仅在云端新cell执行`%pip install torch numpy`后重启runtime。
        """),
        md("## 1. 环境与随机性\n本notebook固定NumPy/Torch seed以便学习；你的原项目只设置了环境随机源，不能反向推断历史训练完全确定。"),
        code("""
        import math
        import platform
        import numpy as np
        import torch
        from torch import nn
        from torch.distributions import Normal
        import torch.nn.functional as F
        SEED = 7
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        print('Python:', platform.python_version(), 'NumPy:', np.__version__, 'Torch:', torch.__version__)
        print('Scope: synthetic tensor/gradient trace on CPU, not Walker training.')
        """),
        md("""
        ## 2. 三个字段怎样拼成24维
        本例B=4，仅为可读的教学batch，不是项目的256 minibatch或4096触发阈值。
        orientations用七个角的cos/sin构造shape；不是实际引擎xmat读数，整组mock也不保证对应真实可实现的状态。
        RunningMeanStd保留项目统计合并逻辑，但这里一次更新一个合成batch；项目rollout逐步更新并存当时normalized obs。
        """),
        code("""
        B = 4
        angles = np.linspace(-.4, .5, B*7, dtype=np.float32).reshape(B, 7)
        orientations = np.stack([np.cos(angles), np.sin(angles)], axis=-1).reshape(B, 14)
        height = np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32)[:, None]
        velocity = np.linspace(-.6, .7, B*9, dtype=np.float32).reshape(B, 9)
        observation = np.concatenate([orientations, height, velocity], axis=-1)
        assert observation.shape == (B, 24)
        class RunningMeanStd:
            # Adapted from the project utility; its original comment credits Gemini assistance.
            def __init__(self, shape):
                self.mean = np.zeros(shape, dtype=np.float32)
                self.var = np.ones(shape, dtype=np.float32)
                self.count = 1e-4
            def update(self, x):
                delta = x.mean(0)-self.mean
                total = self.count+len(x)
                m2 = self.var*self.count+x.var(0)*len(x)+delta**2*self.count*len(x)/total
                self.mean = self.mean+delta*len(x)/total
                self.var, self.count = m2/total, total
            def normalize(self, x):
                return (x-self.mean)/np.sqrt(self.var+1e-8)
        rms = RunningMeanStd((24,))
        rms.update(observation)
        x = torch.tensor(rms.normalize(observation), dtype=torch.float32)
        print('orientation / height / velocity:', orientations.shape, height.shape, velocity.shape)
        print('raw / normalized:', observation.shape, tuple(x.shape))
        assert torch.isfinite(x).all()
        """),
        md("""
        ## 3. Actor输出mu，另有可学习log_std；critic输出V
        网络尺寸、Tanh、初始化与项目对应。fixed_var的含义在这里明确写为state-independent，但参数可学习。
        本例只实现项目实际使用的分支。六维动作是motor controls，hidden width256不是自由度数。
        """),
        code("""
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.mu_net = nn.Sequential(nn.Linear(24, 256), nn.Tanh(),
                                            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 6))
                self.log_std = nn.Parameter(torch.full((6,), -.5))
                for layer in self.mu_net:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2)); nn.init.zeros_(layer.bias)
                nn.init.orthogonal_(self.mu_net[-1].weight, gain=.01)
            def forward(self, obs):
                mu = self.mu_net(obs)
                return mu, self.log_std.exp().expand_as(mu)
        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(24, 256), nn.Tanh(),
                                         nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
                for layer in self.net:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2)); nn.init.zeros_(layer.bias)
                nn.init.orthogonal_(self.net[-1].weight, gain=1.)
            def forward(self, obs):
                return self.net(obs)
        actor, critic = Actor(), Critic()
        mu, std = actor(x)
        value = critic(x)
        print('mu:', tuple(mu.shape), 'std:', tuple(std.shape), 'V:', tuple(value.shape))
        print('shared log_std:', tuple(actor.log_std.shape), 'learnable:', actor.log_std.requires_grad)
        assert mu.shape == std.shape == (B, 6) and value.shape == (B, 1)
        assert torch.allclose(std[0], std[1])
        assert math.isclose(float(std[0, 0].detach()), math.exp(-.5), rel_tol=1e-6)
        """),
        md("""
        ## 4. Raw动作、执行clip与联合log-prob
        同一条记录存raw动作及其old log-prob；项目执行clip后的控制，更新仍评价相同raw动作。
        此处没有真的env.step。演示的饱和动作与policy采样分别保留。
        """),
        code("""
        with torch.no_grad():
            old_mu, old_std = actor(x)
            old_dist = Normal(old_mu, old_std)
            raw_action = old_dist.sample()
            env_action = raw_action.clamp(-1., 1.)
            per_dim_logp = old_dist.log_prob(raw_action)
            old_logp = per_dim_logp.sum(dim=-1)
            old_value = critic(x).squeeze(-1)
        check_mu, check_std = actor(x)
        initial_ratio = (Normal(check_mu, check_std).log_prob(raw_action).sum(-1)-old_logp).exp()
        assert per_dim_logp.shape == (B, 6) and old_logp.shape == (B,)
        assert torch.allclose(initial_ratio, torch.ones(B), atol=1e-6)
        forced_example = torch.tensor([1.3, -1.2, .2, 0., .7, -.4])
        print('raw example:', forced_example.tolist(), 'env:', forced_example.clamp(-1, 1).tolist())
        print('per-dim / joint logp:', tuple(per_dim_logp.shape), tuple(old_logp.shape))
        print('initial ratios:', initial_ratio.detach().tolist())
        print('Six ratios of 1.01 produce joint ratio:', 1.01**6)
        """),
        md("""
        ## 5. GAE：timeout与terminal的末端bootstrap
        下列参考函数使用R6的两种mask，便于解释新环境边界；它不是逐字复制原项目把last统一标timeout的逻辑。
        同一条三步轨迹，只改变末端性质。所有数值是手算例子。
        """),
        code("""
        def gae(rewards, values, final_value, terminated=False, gamma=.99, lam=.95):
            advantage = [0.]*len(rewards)
            carry = 0.
            for t in reversed(range(len(rewards))):
                last = t == len(rewards)-1
                next_v = final_value if last else values[t+1]
                bootstrap = 0. if last and terminated else 1.
                continuation = 0. if last else 1.
                delta = rewards[t]+gamma*bootstrap*next_v-values[t]
                carry = delta+gamma*lam*continuation*carry
                advantage[t] = carry
            returns = [a+v for a, v in zip(advantage, values)]
            return advantage, returns
        a_timeout, r_timeout = gae([1., 1., 1.], [.5, .6, .7], .8)
        a_terminal, r_terminal = gae([1., 1., 1.], [.5, .6, .7], .8, terminated=True)
        assert math.isclose(a_timeout[0], 3.087884453, abs_tol=1e-9)
        assert math.isclose(a_terminal[0], 2.387328575, abs_tol=1e-9)
        print('timeout A / R:', a_timeout, r_timeout)
        print('terminal A / R:', a_terminal, r_terminal)
        """),
        md("""
        ## 6. 一次PPO minibatch更新：仅验证接口和梯度
        将前面的四个mock observation当作一条合成trajectory，使用人为rewards；它们没有经过Walker动力学，不能衡量步态。
        old动作/logp/value固定，先算GAE/targets，再标准化advantage；与项目一样无entropy项。
        实际项目每批10 epochs，本例仅一步，便于观察before/after。初始actor loss近0不表示actor梯度为0。
        """),
        code("""
        assert B == 4, 'If changing B, also change this synthetic reward sequence.'
        teaching_adv, teaching_return = gae([.2, .4, .6, .3], old_value.tolist(), 0., terminated=True)
        advantage = torch.tensor(teaching_adv, dtype=torch.float32)
        returns = torch.tensor(teaching_return, dtype=torch.float32)
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        params = list(actor.parameters())+list(critic.parameters())
        optimizer = torch.optim.Adam(params, lr=3e-4)
        new_mu, new_std = actor(x)
        new_logp = Normal(new_mu, new_std).log_prob(raw_action).sum(-1)
        ratio = (new_logp-old_logp).exp()
        actor_loss = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()
        predicted_value = critic(x).squeeze(-1)
        assert predicted_value.shape == returns.shape == (B,)
        critic_loss = F.mse_loss(predicted_value, returns)
        loss = actor_loss+.5*critic_loss
        optimizer.zero_grad(); loss.backward()
        actor_grad = math.sqrt(sum(float(p.grad.square().sum()) for p in actor.parameters() if p.grad is not None))
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        pre_clip_norm = nn.utils.clip_grad_norm_(params, .5)
        optimizer.step()
        with torch.no_grad():
            post_mu, post_std = actor(x)
            post_ratio = (Normal(post_mu, post_std).log_prob(raw_action).sum(-1)-old_logp).exp()
        print('actor / critic / combined loss:', float(actor_loss.detach()), float(critic_loss.detach()), float(loss.detach()))
        print('actor gradient norm:', actor_grad, 'combined pre-clip norm:', float(pre_clip_norm))
        print('ratios after one optimizer step:', post_ratio.tolist())
        print('One synthetic update completed in this runtime; no Walker performance measured.')
        """),
        md("""
        ## 7. 两个常见错误与项目数字
        `[B,1] - [B]`会广播成矩阵；4096是buffer触发阈值，不是强制裁成4096。
        三种clip分别针对动作、ratio surrogate、梯度，不能混为一种限制。
        """),
        code("""
        with torch.no_grad():
            wrong_shape = (critic(x)-returns).shape
            right_shape = (critic(x).squeeze(-1)-returns).shape
        assert tuple(wrong_shape) == (B, B) and tuple(right_shape) == (B,)
        print('broadcast bug / correct:', tuple(wrong_shape), tuple(right_shape))
        threshold, trajectory_steps, minibatch, epochs = 4096, 1000, 256, 10
        collected = 0
        while collected < threshold:
            collected += trajectory_steps
        full, remainder = divmod(collected, minibatch)
        count = math.ceil(collected/minibatch)
        assert (collected, full, remainder, count*epochs) == (5000, 19, 136, 200)
        print('data:', collected, 'minibatches/epoch:', count, 'last minibatch:', remainder,
              'optimizer steps:', count*epochs)
        print('Clip objects: motor input [-1,1]; PPO surrogate ratio [.8,1.2]; combined gradient norm .5')
        """),
        md("""
        ## 课后记录
        1. 24维的三组字段是什么？为什么不叫24个关节角？
        2. 六维mu、六维std、一个critic值分别代表什么？
        3. 同一个raw动作为什么在新旧policy下计算两个logp？
        4. 末端bootstrap和reset后的GAE连接为什么是两个问题？
        5. loss数值近0为何仍能更新actor？
        6. 本notebook没有验证哪些项目结果？

        保存自己的Colab运行日期、依赖版本、seed、修改项、assert结果和解释，再下载带输出的ipynb。
        本notebook的合成梯度检查不能填写成Walker训练成功，也不能证明历史900+回报的稳定性。
        """),
    ]


def main():
    cells = build()
    for i, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{NAME}:{i}:{cell['source']}".encode()).hexdigest()[:12]
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": NAME, "provenance": []}}, "cells": cells}
    target = ROOT / "notebooks" / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    print("Generated", NAME, "with", sum(c["cell_type"] == "code" for c in cells), "code cells; none executed.")


if __name__ == "__main__":
    main()
