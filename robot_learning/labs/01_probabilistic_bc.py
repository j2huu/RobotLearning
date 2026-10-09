"""Compare conditional means and distributions on a synthetic BC dataset.

This is a CPU teaching experiment, not a robot rollout or paper reproduction.
Run from the workspace root:
    python robot_learning/labs/01_probabilistic_bc.py --steps 800 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        output_dim = {"mse": 1, "gaussian": 2, "mixture": 6}[kind]
        self.body = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.head = nn.Linear(32, output_dim)
        if kind == "mixture":
            # Break component symmetry; neither true branch is supplied as input.
            with torch.no_grad():
                self.head.weight.mul_(0.1)
                self.head.bias.copy_(torch.tensor([0.0, 0.0, -1.0, 1.0, -1.5, -1.5]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs))


def log_normal(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((action - mean) / log_std.exp()).square() - log_std - 0.5 * math.log(2 * math.pi)


def loss(policy: Policy, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    output = policy(obs)
    if policy.kind == "mse":
        return (output - action).square().mean()
    if policy.kind == "gaussian":
        mean, log_std = output[:, :1], output[:, 1:2].clamp(-5.0, 2.0)
        return -log_normal(action, mean, log_std).mean()
    log_weights = output[:, :2].log_softmax(dim=-1)
    means = output[:, 2:4]
    log_stds = output[:, 4:6].clamp(-5.0, 2.0)
    component_log_probs = log_weights + log_normal(action, means, log_stds)
    return -torch.logsumexp(component_log_probs, dim=-1).mean()


def make_data(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(seed)
    n_episodes, episode_length = 64, 64
    obs = torch.linspace(-1.0, 1.0, episode_length).reshape(1, episode_length, 1)
    obs = obs.repeat(n_episodes, 1, 1)
    obs = obs + 0.03 * torch.randn(obs.shape, generator=generator)
    branches = (2 * (torch.arange(n_episodes) % 2) - 1).reshape(n_episodes, 1, 1)
    action = 0.5 * obs + branches + 0.05 * torch.randn(obs.shape, generator=generator)

    # Split balanced episode pairs; no episode appears in both partitions.
    pairs = torch.randperm(n_episodes // 2, generator=generator)
    train_ids = torch.stack((2 * pairs[:24], 2 * pairs[:24] + 1), dim=1).flatten()
    validation_ids = torch.stack((2 * pairs[24:], 2 * pairs[24:] + 1), dim=1).flatten()
    metadata = {
        "n_episodes": n_episodes,
        "episode_length": episode_length,
        "train_episode_ids": train_ids.tolist(),
        "validation_episode_ids": validation_ids.tolist(),
        "true_noise_std": 0.05,
        "observation": "scalar o; branch b is hidden from this single-step policy",
        "expert_action": "a = 0.5 * o + b + Normal(0, 0.05^2), b in {-1, +1}",
        "limitation": "Synthetic episode grouping only; no transition dynamics or robot rollout.",
    }
    return (
        obs[train_ids].reshape(-1, 1),
        action[train_ids].reshape(-1, 1),
        obs[validation_ids].reshape(-1, 1),
        action[validation_ids].reshape(-1, 1),
        metadata,
    )


@torch.no_grad()
def mean_prediction(policy: Policy, obs: torch.Tensor) -> torch.Tensor:
    output = policy(obs)
    if policy.kind == "mse":
        return output
    if policy.kind == "gaussian":
        return output[:, :1]
    return (output[:, :2].softmax(dim=-1) * output[:, 2:4]).sum(dim=-1, keepdim=True)


@torch.no_grad()
def sample_actions(policy: Policy, obs: torch.Tensor) -> torch.Tensor:
    output = policy(obs)
    if policy.kind == "mse":
        return output
    if policy.kind == "gaussian":
        return output[:, :1] + output[:, 1:2].clamp(-5.0, 2.0).exp() * torch.randn_like(obs)
    components = torch.multinomial(output[:, :2].softmax(dim=-1), num_samples=1)
    means = output[:, 2:4].gather(1, components)
    stds = output[:, 4:6].clamp(-5.0, 2.0).exp().gather(1, components)
    return means + stds * torch.randn_like(obs)


@torch.no_grad()
def evaluate(policy: Policy, obs: torch.Tensor, action: torch.Tensor) -> dict:
    policy.eval()
    mean = mean_prediction(policy, obs)
    sample_obs = 2 * torch.rand(8192, 1) - 1
    samples = sample_actions(policy, sample_obs)
    center = 0.5 * sample_obs
    distance = torch.minimum((samples - (center - 1)).abs(), (samples - (center + 1)).abs())
    output_zero = policy(torch.zeros(1, 1))
    if policy.kind == "mse":
        at_zero = {"mean": output_zero.item()}
    elif policy.kind == "gaussian":
        at_zero = {
            "mean": output_zero[0, 0].item(),
            "std": output_zero[0, 1].clamp(-5.0, 2.0).exp().item(),
        }
    else:
        at_zero = {
            "weights": output_zero[0, :2].softmax(dim=-1).tolist(),
            "means": output_zero[0, 2:4].tolist(),
            "stds": output_zero[0, 4:6].clamp(-5.0, 2.0).exp().tolist(),
        }
    return {
        "held_out_mean_prediction_mse": (mean - action).square().mean().item(),
        "held_out_nll": None if policy.kind == "mse" else loss(policy, obs, action).item(),
        "sampled_mode_compatibility": (distance < 0.15).float().mean().item(),
        "mode_compatibility_threshold": 0.15,
        "sampled_positive_branch_fraction": (samples > center).float().mean().item(),
        "distribution_at_o_zero": at_zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_obs, train_action, validation_obs, validation_action, metadata = make_data(args.seed)
    result = {
        "config": {"seed": args.seed, "steps": args.steps, "batch_size": 128, "lr": 0.003, "device": "cpu"},
        "data": metadata,
        "models": {},
        "metric_note": "Mode compatibility measures closeness to synthetic expert modes, not robot task success.",
    }
    for index, kind in enumerate(("mse", "gaussian", "mixture")):
        torch.manual_seed(args.seed + 100 * index)
        policy = Policy(kind)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
        policy.train()
        for _ in range(args.steps):
            indices = torch.randint(len(train_obs), (128,))
            objective = loss(policy, train_obs[indices], train_action[indices])
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        result["models"][kind] = evaluate(policy, validation_obs, validation_action)

    print("model      mean MSE    held-out NLL    sampled mode compatibility")
    for kind, metrics in result["models"].items():
        nll = metrics["held_out_nll"]
        nll_text = "n/a" if nll is None else f"{nll:.4f}"
        print(f"{kind:9s}  {metrics['held_out_mean_prediction_mse']:8.4f}    {nll_text:>12s}    {metrics['sampled_mode_compatibility']:.3f}")
        print("  o=0:", json.dumps(metrics["distribution_at_o_zero"], ensure_ascii=False))
    print(result["metric_note"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print("Saved:", args.output)


if __name__ == "__main__":
    main()
