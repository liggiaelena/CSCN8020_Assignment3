from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent, set_global_seed
from g1_rl import G1ElbowTargetEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = (
    REPO_ROOT
    / "assets"
    / "g1_fixed_base"
    / "scene_29dof_fixed_base.xml"
)

# Required benchmark goals (section 3.4): four angles, five
# episodes each, twenty evaluation episodes total.
BENCHMARK_GOALS = [-0.8, -0.4, 0.4, 0.8]
EPISODES_PER_GOAL = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedily evaluate a trained DQN checkpoint "
        "on the four required benchmark goals."
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Defaults to <checkpoint-dir>/../results/"
        "<checkpoint-stem>_evaluation.csv",
    )

    return parser.parse_args()


def evaluate_goal(
    env: G1ElbowTargetEnv,
    agent: DQNAgent,
    goal_angle: float,
    episodes: int,
    action_rng: np.random.Generator,
) -> list[dict[str, float]]:
    rows = []

    for episode_index in range(episodes):
        observation, info = env.reset(
            options={"goal_angle": goal_angle},
        )

        cumulative_reward = 0.0

        while True:
            # epsilon = 0.0: fully greedy, no exploration, per the
            # assignment's evaluation requirement.
            action = agent.select_action(
                observation,
                epsilon=0.0,
                rng=action_rng,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            cumulative_reward += reward

            if terminated or truncated:
                break

        rows.append(
            {
                "goal_angle": goal_angle,
                "episode_index": episode_index,
                "success": bool(info["is_success"]),
                "cumulative_reward": cumulative_reward,
                "episode_length": int(info["episode_step"]),
                "final_absolute_error": float(
                    info["absolute_error"]
                ),
            }
        )

    return rows


def main() -> None:
    args = parse_arguments()

    set_global_seed(args.seed)
    action_rng = np.random.default_rng(args.seed)

    env = G1ElbowTargetEnv(
        scene_path=SCENE_PATH,
        maximum_episode_steps=args.max_episode_steps,
    )

    device = torch.device("cpu")

    agent = DQNAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    )
    agent.load_checkpoint(args.checkpoint, map_location=device)

    checkpoint_stem = Path(args.checkpoint).stem
    output_csv = Path(
        args.output_csv
        or (
            REPO_ROOT
            / "results"
            / f"{checkpoint_stem}_evaluation.csv"
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float]] = []

    try:
        for goal_angle in BENCHMARK_GOALS:
            goal_rows = evaluate_goal(
                env=env,
                agent=agent,
                goal_angle=goal_angle,
                episodes=EPISODES_PER_GOAL,
                action_rng=action_rng,
            )
            all_rows.extend(goal_rows)

            successes = sum(row["success"] for row in goal_rows)
            mean_reward = float(
                np.mean(
                    [row["cumulative_reward"] for row in goal_rows]
                )
            )

            print(
                f"goal={goal_angle:+.2f} rad | "
                f"successes={successes}/{EPISODES_PER_GOAL} | "
                f"mean_reward={mean_reward:+.3f}"
            )
    finally:
        env.close()

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    overall_successes = sum(row["success"] for row in all_rows)
    overall_success_rate = overall_successes / len(all_rows)

    print(
        f"\nOverall: {overall_successes}/{len(all_rows)} "
        f"successes ({overall_success_rate:.1%})"
    )
    print(f"Results written to: {output_csv}")


if __name__ == "__main__":
    main()
