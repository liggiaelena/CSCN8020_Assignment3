from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = (
    REPO_ROOT
    / "assets"
    / "g1_fixed_base"
    / "scene_29dof_fixed_base.xml"
)

# Same benchmark protocol as evaluate_dqn.py, so the two policies
# are compared on identical goals and episode counts (section 7).
BENCHMARK_GOALS = [-0.8, -0.4, 0.4, 0.8]
EPISODES_PER_GOAL = 5
ACTION_INCREMENT = 0.08


def evaluate_goal(
    env: G1ElbowTargetEnv,
    goal_angle: float,
    episodes: int,
) -> list[dict[str, float]]:
    rows = []

    for episode_index in range(episodes):
        observation, info = env.reset(
            options={"goal_angle": goal_angle},
        )

        cumulative_reward = 0.0

        while True:
            action = choose_rule_based_action(
                observation=observation,
                controller_target=float(info["controller_target"]),
                action_increment=ACTION_INCREMENT,
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate choose_rule_based_action() on the "
        "same four benchmark goals used for the DQN evaluation."
    )
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument(
        "--output-csv",
        default=str(
            REPO_ROOT / "results" / "rule_based_evaluation.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    env = G1ElbowTargetEnv(
        scene_path=SCENE_PATH,
        maximum_episode_steps=args.max_episode_steps,
    )

    all_rows: list[dict[str, float]] = []

    try:
        for goal_angle in BENCHMARK_GOALS:
            goal_rows = evaluate_goal(env, goal_angle, EPISODES_PER_GOAL)
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

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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
