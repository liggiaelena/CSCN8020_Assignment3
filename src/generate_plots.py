from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_training_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    columns = {
        key: np.array(
            [row[key] for row in rows],
            dtype=float,
        )
        for key in rows[0]
        if key != "success"
    }
    columns["success"] = np.array(
        [row["success"] == "True" for row in rows],
        dtype=float,
    )

    return columns


def read_evaluation_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan)

    kernel = np.ones(window) / window

    padded = np.concatenate(
        [np.full(window - 1, values[0]), values]
    )

    return np.convolve(padded, kernel, mode="valid")


def plot_training_reward(
    configs: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in configs.items():
        episodes = data["episode"]
        reward = data["cumulative_reward"]

        ax.plot(
            episodes,
            reward,
            alpha=0.25,
            linewidth=0.8,
        )
        ax.plot(
            episodes,
            moving_average(reward, 20),
            label=f"{label} (20-episode moving average)",
            linewidth=2,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Training reward: raw and moving average")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_success_rate(
    configs: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    window: int = 50,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in configs.items():
        rolling_success = moving_average(data["success"], window)

        ax.plot(
            data["episode"],
            rolling_success,
            label=label,
            linewidth=2,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success rate ({window}-episode rolling window)")
    ax.set_title("Training success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_epsilon(
    configs: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in configs.items():
        ax.plot(
            data["episode"],
            data["epsilon"],
            label=label,
            linewidth=2,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration-rate decay")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loss(
    configs: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in configs.items():
        loss = data["mean_loss"]
        valid = ~np.isnan(loss)

        ax.plot(
            data["episode"][valid],
            loss[valid],
            label=label,
            linewidth=1.2,
            alpha=0.8,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Huber loss (per episode)")
    ax.set_title("Optimization loss over training")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_evaluation_success_by_goal(
    evaluation_rows: list[dict[str, str]],
    label: str,
    output_path: Path,
) -> None:
    goals = sorted(
        {float(row["goal_angle"]) for row in evaluation_rows}
    )

    success_rates = []

    for goal in goals:
        rows_for_goal = [
            row
            for row in evaluation_rows
            if float(row["goal_angle"]) == goal
        ]
        successes = sum(
            row["success"] == "True" for row in rows_for_goal
        )
        success_rates.append(successes / len(rows_for_goal))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        [f"{g:+.1f}" for g in goals],
        success_rates,
        color="#4C72B0",
    )
    ax.axhline(
        0.8,
        color="red",
        linestyle="--",
        label="80% required threshold",
    )
    ax.set_xlabel("Benchmark goal angle (rad)")
    ax.set_ylabel("Success rate")
    ax.set_title(f"Evaluation success rate by goal -- {label}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_config_comparison_table(
    configs: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8") as table_file:
        table_file.write(
            "config,total_episodes,wall_clock_minutes,"
            "final_epsilon,mean_reward_final_20,"
            "success_rate_final_50\n"
        )

        for label, data in configs.items():
            total_episodes = int(data["episode"][-1])
            wall_clock_minutes = (
                data["elapsed_seconds"][-1] / 60.0
            )
            final_epsilon = data["epsilon"][-1]
            mean_reward_final_20 = float(
                np.mean(data["cumulative_reward"][-20:])
            )
            success_rate_final_50 = float(
                np.mean(data["success"][-50:])
            )

            table_file.write(
                f"{label},{total_episodes},"
                f"{wall_clock_minutes:.3f},{final_epsilon:.4f},"
                f"{mean_reward_final_20:.4f},"
                f"{success_rate_final_50:.4f}\n"
            )

    print(f"Config comparison table written to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the required training/evaluation "
        "plots and the epsilon-decay comparison table."
    )

    parser.add_argument(
        "--config-a-csv",
        default=str(
            REPO_ROOT / "results" / "config_a" / "training_metrics.csv"
        ),
    )
    parser.add_argument(
        "--config-b-csv",
        default=str(
            REPO_ROOT / "results" / "config_b" / "training_metrics.csv"
        ),
    )
    parser.add_argument(
        "--selected-config-eval-csv",
        required=True,
        help="Evaluation CSV of the config chosen as the final DQN.",
    )
    parser.add_argument(
        "--selected-config-label",
        default="Selected DQN",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "plots"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "Config A (decay=0.995)": read_training_csv(
            Path(args.config_a_csv)
        ),
        "Config B (decay=0.985)": read_training_csv(
            Path(args.config_b_csv)
        ),
    }

    plot_training_reward(configs, output_dir / "training_reward.png")
    plot_success_rate(configs, output_dir / "training_success_rate.png")
    plot_epsilon(configs, output_dir / "epsilon_decay.png")
    plot_loss(configs, output_dir / "training_loss.png")

    write_config_comparison_table(
        configs,
        output_dir / "config_comparison.csv",
    )

    evaluation_rows = read_evaluation_csv(
        Path(args.selected_config_eval_csv)
    )
    plot_evaluation_success_by_goal(
        evaluation_rows,
        args.selected_config_label,
        output_dir / "evaluation_success_by_goal.png",
    )

    print(f"All plots written to: {output_dir}")


if __name__ == "__main__":
    main()
