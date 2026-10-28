"""Check flow conventions, attention leakage and delay arithmetic with stdlib.

This deterministic teaching lab uses known endpoints and hand-built attention.
It does not train a model, implement openpi, or simulate robot dynamics.
Run from the repository root; --output is optional.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def max_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Compared vectors must have equal lengths.")
    return max(abs(a - b) for a, b in zip(left, right))


def check_flow(seed: int, steps: int) -> dict:
    rng = random.Random(seed)
    shape = [2, 4, 3]
    count = math.prod(shape)
    actions = [rng.uniform(-2.0, 2.0) for _ in range(count)]
    noise = [rng.gauss(0.0, 1.0) for _ in range(count)]
    target_forward = [a - e for a, e in zip(actions, noise)]
    target_reverse = [-u for u in target_forward]
    tau, h = 0.37, 1e-6

    def path(t: float) -> list[float]:
        return [(1 - t) * e + t * a for a, e in zip(actions, noise)]

    finite_difference = [(p - m) / (2 * h) for p, m in zip(path(tau + h), path(tau - h))]

    def integrate(velocity: list[float], dt: float) -> list[float]:
        # This is an endpoint-conditioned oracle, not an inference-time model.
        x = noise[:]
        for _ in range(steps):
            x = [value + dt * speed for value, speed in zip(x, velocity)]
        return x

    forward = integrate(target_forward, 1.0 / steps)
    reverse = integrate(target_reverse, -1.0 / steps)
    wrong_sign = integrate(target_reverse, 1.0 / steps)
    expected_wrong = [2 * e - a for a, e in zip(actions, noise)]
    derivative_error = max_difference(finite_difference, target_forward)
    forward_error = max_difference(forward, actions)
    reverse_error = max_difference(reverse, actions)
    convention_difference = max_difference(forward, reverse)
    wrong_error = max_difference(wrong_sign, actions)
    if derivative_error > 1e-8 or max(forward_error, reverse_error, convention_difference) > 1e-10:
        raise AssertionError("Consistent flow conventions did not agree with the path.")
    if wrong_error < 1e-3 or max_difference(wrong_sign, expected_wrong) > 1e-10:
        raise AssertionError("The deliberately inconsistent target/dt did not fail as expected.")
    return {
        "conceptual_tensor_shape": shape,
        "finite_difference_max_error": derivative_error,
        "forward_endpoint_max_error": forward_error,
        "reverse_endpoint_max_error": reverse_error,
        "convention_max_difference": convention_difference,
        "wrong_sign_endpoint_max_error": wrong_error,
        "limitation": "Known endpoints and constant oracle velocity; no learned generative policy.",
    }


def build_mask(leak_prefix: bool) -> list[list[bool]]:
    groups = ["P", "P", "D", "D", "C", "C", "PAD"]
    valid = [True, True, True, True, True, True, False]
    mask = []
    for i, query in enumerate(groups):
        row = []
        for j, key in enumerate(groups):
            allowed = (
                (query == "P" and (key == "P" or (leak_prefix and key == "D")))
                or (query == "D" and (key == "P" or (key == "D" and j <= i)))
                or (query == "C" and key in {"P", "C"})
            )
            row.append(valid[i] and valid[j] and allowed)
        mask.append(row)
    return mask


def attention_layer(values: list[list[float]], mask: list[list[bool]]) -> list[list[float]]:
    """One scalar-head residual layer with Q=K=V for a transparent counterexample."""
    width = len(values[0])
    output = []
    for query, row in zip(values, mask):
        indices = [j for j, allowed in enumerate(row) if allowed]
        if not indices:
            output.append(query[:])
            continue
        scores = [sum(q * k for q, k in zip(query, values[j])) / math.sqrt(width) for j in indices]
        largest = max(scores)
        weights = [math.exp(score - largest) for score in scores]
        denominator = sum(weights)
        aggregate = [
            sum(weight * values[j][d] for weight, j in zip(weights, indices)) / denominator
            for d in range(width)
        ]
        output.append([q + a for q, a in zip(query, aggregate)])
    return output


def action_difference(left: list[list[float]], right: list[list[float]]) -> float:
    return max_difference([x for row in left[4:6] for x in row], [x for row in right[4:6] for x in row])


def check_attention() -> dict:
    initial = [[0.1, 0.2], [0.2, -0.1], [0.3, 0.4], [-0.2, 0.1], [0.1, -0.3], [0.4, 0.2], [9.0, 9.0]]
    perturbed = [row[:] for row in initial]
    perturbed[2] = [2.3, 1.4]  # Change clean FAST action targets only.
    correct_mask, leaky_mask = build_mask(False), build_mask(True)

    def twice(data: list[list[float]], mask: list[list[bool]]) -> list[list[float]]:
        return attention_layer(attention_layer(data, mask), mask)

    correct_delta = action_difference(twice(initial, correct_mask), twice(perturbed, correct_mask))
    leaky_one_delta = action_difference(attention_layer(initial, leaky_mask), attention_layer(perturbed, leaky_mask))
    leaky_two_delta = action_difference(twice(initial, leaky_mask), twice(perturbed, leaky_mask))
    padded = [row[:] for row in initial]
    padded[-1] = [-100.0, 100.0]
    pad_delta = action_difference(twice(initial, correct_mask), twice(padded, correct_mask))
    future_discrete = [row[:] for row in initial]
    future_discrete[3] = [3.0, -2.0]
    causal_delta = max_difference(twice(initial, correct_mask)[2], twice(future_discrete, correct_mask)[2])
    if max(correct_delta, leaky_one_delta, pad_delta, causal_delta) > 1e-12:
        raise AssertionError("A forbidden direct or indirect dependency leaked through the correct mask.")
    if leaky_two_delta < 1e-3:
        raise AssertionError("The deliberate D -> P -> C leakage was not detected.")
    return {
        "token_order": ["P0", "P1", "D0", "D1", "C0", "C1", "padding"],
        "correct_mask": [[int(v) for v in row] for row in correct_mask],
        "correct_two_layer_action_delta": correct_delta,
        "leaky_one_layer_action_delta": leaky_one_delta,
        "leaky_two_layer_action_delta": leaky_two_delta,
        "padding_perturbation_action_delta": pad_delta,
        "future_FAST_perturbation_past_FAST_delta": causal_delta,
        "limitation": "Hand-built identity Q/K/V attention; no trained Transformer or gradient routing test.",
    }


def delay_case(latency_ms: float, action_hz: int, execute_steps: int, horizon: int) -> dict:
    elapsed_intervals = math.ceil(latency_ms * action_hz / 1000.0)
    window_ms = 1000.0 * execute_steps / action_hz
    return {
        "latency_ms": latency_ms,
        "action_hz": action_hz,
        "execute_steps": execute_steps,
        "horizon": horizon,
        "execution_window_ms": window_ms,
        "async_budget_margin_ms_ignoring_other_overheads": window_ms - latency_ms,
        "elapsed_action_intervals_at_arrival": elapsed_intervals,
        "remaining_waypoints_after_elapsed_prefix": max(0, horizon - elapsed_intervals),
        "synchronous_cycle_ms": window_ms + latency_ms,
        "assumption": "Observation at time zero; inference overlaps execution only in async budget calculation; zero other overhead.",
    }


def check_delay() -> dict:
    cases = [delay_case(ms, 50, 10, 50) for ms in (0, 120, 260, 1200)]
    if cases[1]["elapsed_action_intervals_at_arrival"] != 6:
        raise AssertionError("120 ms at 50 Hz should span six intervals.")
    if cases[2]["async_budget_margin_ms_ignoring_other_overheads"] >= 0:
        raise AssertionError("260 ms inference should miss a 200 ms overlap budget.")
    if cases[3]["remaining_waypoints_after_elapsed_prefix"] != 0:
        raise AssertionError("An expired chunk must have no remaining waypoints.")
    return {
        "cases": cases,
        "limitation": "Clock arithmetic, not a scheduler, physical simulation, or recommendation to blindly discard contact actions.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("--steps must be between 1 and 10000")
    if args.output is not None:
        # This repository requires explicit permission before writing elsewhere.
        repository = Path(__file__).resolve().parents[2]
        if not args.output.resolve().is_relative_to(repository):
            parser.error("--output must stay inside this repository")
    result = {
        "config": {"seed": args.seed, "steps": args.steps, "dependencies": "Python 3.9+ standard library"},
        "flow": check_flow(args.seed, args.steps),
        "attention": check_attention(),
        "delay": check_delay(),
        "all_checks_passed": True,
        "scope": "Mechanism checks only. No policy training, checkpoint, robot rollout, or paper success-rate reproduction.",
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print("Saved:", args.output)


if __name__ == "__main__":
    main()
