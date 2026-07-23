from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent, ReplayBuffer, set_global_seed
from g1_rl import G1ElbowTargetEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = (
    REPO_ROOT
    / "assets"
    / "g1_fixed_base"
    / "scene_29dof_fixed_base.xml"
)

TRAINING_GOAL_RANGE = (-0.8, 0.8)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on G1ElbowTargetEnv."
    )

    parser.add_argument(
        "--config-name",
        required=True,
        help="Label used for output files, e.g. 'config_a'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=50_000,
    )
    parser.add_argument("--initial-epsilon", type=float, default=1.0)
    parser.add_argument("--minimum-epsilon", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        required=True,
        help="Multiplicative epsilon decay applied once per episode.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=250,
        help="Optimizer steps between target-network syncs.",
    )
    parser.add_argument(
        "--warmup-transitions",
        type=int,
        default=500,
        help="Minimum replay-buffer size before optimization starts.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Hard episode cap (stop condition 1 of 2).",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=60.0,
        help="Wall-clock training budget in minutes "
        "(stop condition 2 of 2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to results/<config-name>/.",
    )

    return parser.parse_args()


def run_episode(
    env: G1ElbowTargetEnv,
    agent: DQNAgent,
    replay_buffer: ReplayBuffer,
    epsilon: float,
    batch_size: int,
    warmup_transitions: int,
    action_rng: np.random.Generator,
) -> dict[str, float]:
    observation, info = env.reset()

    cumulative_reward = 0.0
    losses: list[float] = []
    terminated = False
    truncated = False

    while True:
        action = agent.select_action(
            observation,
            epsilon,
            action_rng,
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # Only `terminated` gates bootstrapping -- see
        # dqn/replay_buffer.py for why truncation must not.
        replay_buffer.push(
            observation,
            action,
            reward,
            next_observation,
            terminated,
        )

        observation = next_observation
        cumulative_reward += reward

        if len(replay_buffer) >= warmup_transitions:
            loss = agent.optimize_model(replay_buffer, batch_size)

            if loss is not None:
                losses.append(loss)

        if terminated or truncated:
            break

    return {
        "cumulative_reward": cumulative_reward,
        "success": bool(info["is_success"]),
        "episode_length": int(info["episode_step"]),
        "final_absolute_error": float(info["absolute_error"]),
        "mean_loss": (
            float(np.mean(losses)) if losses else float("nan")
        ),
    }


def main() -> None:
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
        or (REPO_ROOT / "results" / args.config_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    action_rng = np.random.default_rng(args.seed)

    env = G1ElbowTargetEnv(
        scene_path=SCENE_PATH,
        goal_range=TRAINING_GOAL_RANGE,
        maximum_episode_steps=args.max_episode_steps,
    )

    device = torch.device("cpu")

    agent = DQNAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        target_update_interval=args.target_update_interval,
        device=device,
    )

    replay_buffer = ReplayBuffer(args.replay_capacity)

    epsilon = args.initial_epsilon

    # Seed the environment's own RNG once. Every later reset()
    # continues drawing from that same seeded sequence, so the full
    # run of sampled goal angles is reproducible from --seed alone.
    observation, _ = env.reset(seed=args.seed)

    metrics_path = output_dir / "training_metrics.csv"
    fieldnames = [
        "episode",
        "cumulative_reward",
        "success",
        "episode_length",
        "final_absolute_error",
        "epsilon",
        "mean_loss",
        "elapsed_seconds",
    ]

    start_time = time.monotonic()

    try:
        with open(
            metrics_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            episode = 0

            while True:
                episode += 1

                episode_stats = run_episode(
                    env=env,
                    agent=agent,
                    replay_buffer=replay_buffer,
                    epsilon=epsilon,
                    batch_size=args.batch_size,
                    warmup_transitions=args.warmup_transitions,
                    action_rng=action_rng,
                )

                elapsed_seconds = time.monotonic() - start_time

                writer.writerow(
                    {
                        "episode": episode,
                        "epsilon": epsilon,
                        "elapsed_seconds": elapsed_seconds,
                        **episode_stats,
                    }
                )
                metrics_file.flush()

                if episode % 20 == 0 or episode == 1:
                    print(
                        f"episode={episode:5d} | "
                        f"epsilon={epsilon:.3f} | "
                        f"reward={episode_stats['cumulative_reward']:+8.3f} | "
                        f"success={episode_stats['success']} | "
                        f"len={episode_stats['episode_length']:3d} | "
                        f"elapsed={elapsed_seconds / 60.0:6.2f} min"
                    )

                epsilon = max(
                    args.minimum_epsilon,
                    epsilon * args.epsilon_decay,
                )

                elapsed_minutes = elapsed_seconds / 60.0

                if episode >= args.max_episodes:
                    print(
                        f"Stopping: reached max-episodes="
                        f"{args.max_episodes}."
                    )
                    break

                if elapsed_minutes >= args.max_minutes:
                    print(
                        f"Stopping: reached max-minutes="
                        f"{args.max_minutes}."
                    )
                    break

        checkpoint_path = (
            models_dir / f"{args.config_name}_dqn.pt"
        )
        agent.save_checkpoint(checkpoint_path)

        total_minutes = (time.monotonic() - start_time) / 60.0

        print(f"Total episodes: {episode}")
        print(f"Total wall-clock time: {total_minutes:.2f} minutes")
        print(f"Final epsilon: {epsilon:.4f}")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Metrics saved to: {metrics_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
