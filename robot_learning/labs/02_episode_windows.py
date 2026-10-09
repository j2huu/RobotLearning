"""Build causal, masked training windows from synthetic episodes.

This demonstrates data contracts, not a physical robot or a trained policy.
Run from the RobotLearning workspace root:
    python robot_learning/labs/02_episode_windows.py --output robot_learning/labs/results/lab02_episode_windows.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch


@dataclass
class Episode:
    episode_id: int
    time: torch.Tensor
    proprio: torch.Tensor
    action: torch.Tensor
    force_available_time: torch.Tensor
    force: torch.Tensor


def make_episode(episode_id: int, length: int) -> Episode:
    time = torch.arange(length, dtype=torch.float64) / 10.0
    proprio = torch.stack((time + 0.2 * episode_id, torch.sin(time)), dim=-1).float()
    action = proprio + torch.tensor([0.1, -0.05])
    # Force is available every 20 ms, with an offset relative to the 10 Hz steps.
    force_time = torch.arange(length * 5, dtype=torch.float64) / 50.0 - 0.015
    force = (2 * torch.sin(3 * force_time) + episode_id).float().unsqueeze(-1)
    return Episode(episode_id, time, proprio, action, force_time, force)


def aligned_observation(episode: Episode) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.searchsorted(episode.force_available_time, episode.time, right=True) - 1
    if (indices < 0).any():
        raise ValueError("Some control steps have no available force sample.")
    selected_time = episode.force_available_time[indices]
    if (selected_time > episode.time).any():
        raise ValueError("Future force leakage.")
    return torch.cat((episode.proprio, episode.force[indices]), dim=-1), selected_time


def make_window(episode: Episode, step: int, observation_horizon: int, action_horizon: int) -> dict:
    observation, force_time = aligned_observation(episode)
    history_indices = torch.arange(step - observation_horizon + 1, step + 1)
    observation_mask = history_indices >= 0
    safe_history = history_indices.clamp(min=0)
    action_indices = torch.arange(step, step + action_horizon)
    action_mask = action_indices < len(episode.time)
    safe_action = action_indices.clamp(max=len(episode.time) - 1)
    chunk = episode.action[safe_action].clone()
    chunk[~action_mask] = 0
    return {
        "episode_id": episode.episode_id,
        "step": step,
        "observation": observation[safe_history],
        "observation_mask": observation_mask,
        "observation_time": episode.time[safe_history],
        "force_available_time": force_time[safe_history],
        "action": chunk,
        "action_mask": action_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    episodes = [make_episode(i, length) for i, length in enumerate((9, 13, 7, 11, 8, 12))]
    train, validation = episodes[:4], episodes[4:]
    train_ids = {e.episode_id for e in train}
    validation_ids = {e.episode_id for e in validation}
    if train_ids & validation_ids:
        raise ValueError("Episode split leakage.")

    # Each genuine training action contributes once; validation and padding do not.
    train_actions = torch.cat([episode.action for episode in train])
    action_mean = train_actions.mean(dim=0)
    action_std = train_actions.std(dim=0, correction=0).clamp(min=1e-6)
    all_actions = torch.cat([episode.action for episode in episodes])

    observation_horizon, action_horizon = 3, 4
    windows = [
        make_window(episode, step, observation_horizon, action_horizon)
        for episode in train
        for step in range(len(episode.time))
    ]
    observation = torch.stack([window["observation"] for window in windows])
    observation_mask = torch.stack([window["observation_mask"] for window in windows])
    action = torch.stack([window["action"] for window in windows])
    action_mask = torch.stack([window["action_mask"] for window in windows])
    observation_time = torch.stack([window["observation_time"] for window in windows])
    force_time = torch.stack([window["force_available_time"] for window in windows])
    force_age = (observation_time - force_time)[observation_mask]

    normalized = (action - action_mean) / action_std
    normalized[~action_mask] = 0
    recovered = normalized * action_std + action_mean
    round_trip_error = (recovered[action_mask] - action[action_mask]).abs().max().item()
    if round_trip_error > 1e-6:
        raise ValueError("Action normalization round trip failed.")

    # A zero prediction illustrates the masked reduction, without training a model.
    element_loss = normalized.square()
    valid_elements = int(action_mask.sum()) * action.shape[-1]
    masked_loss = (element_loss * action_mask.unsqueeze(-1)).sum() / valid_elements

    result = {
        "config": {"observation_horizon": observation_horizon, "action_horizon": action_horizon},
        "data_contract": {
            "observation": "two synthetic proprio values plus one scalar force; causal availability times",
            "action": "two synthetic continuous target coordinates, not a real robot action interface",
            "padding": "repeat-first history with mask; zero action padding with mask",
            "split": "episode IDs; train and validation only in this teaching example",
            "limitation": "No physical dynamics, learned policy, frame transforms or robot success evaluation.",
        },
        "train_episode_ids": sorted(train_ids),
        "validation_episode_ids": sorted(validation_ids),
        "n_train_actions_for_statistics": len(train_actions),
        "shapes": {"observation": list(observation.shape), "action": list(action.shape)},
        "action_normalization": {
            "train_mean": action_mean.tolist(),
            "train_std": action_std.tolist(),
            "all_data_mean_for_comparison_only": all_actions.mean(dim=0).tolist(),
            "max_valid_round_trip_error": round_trip_error,
        },
        "invalid_history_slots": int((~observation_mask).sum()),
        "invalid_action_slots": int((~action_mask).sum()),
        "force_age_seconds": {"min": force_age.min().item(), "max": force_age.max().item()},
        "future_force_sample_count": int(((force_time > observation_time) & observation_mask).sum()),
        "masked_normalized_mse_of_zero_prediction": masked_loss.item(),
        "boundary_examples": [],
    }
    for window in (windows[0], windows[-1]):
        result["boundary_examples"].append({
            "episode_id": window["episode_id"],
            "step": window["step"],
            "observation_mask": window["observation_mask"].tolist(),
            "action_mask": window["action_mask"].tolist(),
            "action_chunk": window["action"].tolist(),
        })
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("Saved:", args.output)


if __name__ == "__main__":
    main()
