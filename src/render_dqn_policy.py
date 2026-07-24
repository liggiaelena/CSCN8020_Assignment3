from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent
from g1_rl import G1ElbowTargetEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = (
    REPO_ROOT
    / "assets"
    / "g1_fixed_base"
    / "scene_29dof_fixed_base.xml"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the trained DQN policy in the MuJoCo "
        "viewer for the demonstration video. Loads a saved "
        "checkpoint -- it does not train."
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--goals",
        type=float,
        nargs="+",
        default=[-0.8, 0.8],
        help="Target angles to demonstrate in sequence.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument(
        "--pause-between-episodes",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def run_episode(
    env: G1ElbowTargetEnv,
    agent: DQNAgent,
    goal_angle: float,
    action_rng: np.random.Generator,
) -> None:
    observation, info = env.reset(
        options={"goal_angle": goal_angle},
    )

    print(f"\n=== GOAL {goal_angle:+.2f} rad ===")

    while True:
        # epsilon = 0.0: this must be the greedy learned policy,
        # never the rule-based one, per the video requirements.
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

        print(
            f"step={info['episode_step']:3d} | "
            f"action={action} | "
            f"angle={info['elbow_angle']:+.4f} | "
            f"goal={info['goal_angle']:+.4f} | "
            f"error={info['angle_error']:+.4f} | "
            f"reward={reward:+.4f}"
        )

        if terminated or truncated:
            break

    print(
        f"Episode result: "
        f"{'SUCCESS' if info['is_success'] else 'FAILURE'} "
        f"(final error={info['absolute_error']:.4f}, "
        f"steps={info['episode_step']})"
    )


def main() -> None:
    args = parse_arguments()

    action_rng = np.random.default_rng(args.seed)

    env = G1ElbowTargetEnv(
        scene_path=SCENE_PATH,
        render_mode="human",
        maximum_episode_steps=args.max_episode_steps,
    )

    device = torch.device("cpu")

    agent = DQNAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    )
    agent.load_checkpoint(args.checkpoint, map_location=device)

    env.render()

    print("Starting in 7 seconds...")
    time.sleep(8.0)

    try:
        for goal_angle in args.goals:
            run_episode(env, agent, goal_angle, action_rng)
            time.sleep(args.pause_between_episodes)

        print("\nDemonstration complete. Close the viewer to exit.")

        while env.viewer is not None and env.viewer.is_running():
            time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
