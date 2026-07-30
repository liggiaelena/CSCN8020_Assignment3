# DQN Control of the Unitree G1 Left Elbow

**Assignment:** CSCN8020 DQN Assignment -- Deep Q-Network Control of the Unitree G1 Left Elbow
**Student:** Liggia Cruz
**Student ID:** 9085905
**Instructor:** Prof. Enrique Espinosa, Conestoga College
**Repository:** https://github.com/liggiaelena/CSCN8020_Assignment3
**Clone URL:** https://github.com/liggiaelena/CSCN8020_Assignment3.git
**Validated run environment:** Python 3.12, WSL2 Ubuntu 24.04 (exact package versions pinned in `requirements-lock.txt`)

---

## Results at a Glance

The selected DQN (**Config B**, faster epsilon decay = 0.985) meets the assignment's
required 80% benchmark-success threshold with a perfect **100% (20/20)** result,
matching the hand-written rule-based baseline while reaching the success streak
in fewer steps on average.

| Metric | Rule-based policy | Selected DQN (Config B) |
|---|---|---|
| Successes / 20 | 20 | 20 |
| Success rate | 100% | 100% |
| Mean cumulative reward | 12.87 | 13.11 |
| Mean episode length | 24.0 steps | 19.5 steps |
| Mean final absolute error | 0.0122 rad | 0.0123 rad |

Both epsilon-decay configurations (A = 0.995, B = 0.985) reached 100% benchmark
success; Config B was selected because it reached a 90%+ rolling training
success rate sooner and ended training with slightly lower reward variance.
Full per-goal breakdown, training curves, and discussion:
`report/DQN_Assignment_Report.pdf`.

**See it in action:**
- Demonstration video (selected DQN, epsilon = 0.0, two target angles): https://drive.google.com/file/d/1Re9CCpDgAQ9mob4pyjiAd0DNBoDP6R2G/view?usp=sharing
  (backup folder if that link doesn't open: https://drive.google.com/drive/folders/1jCXgg2mqQP1JkAB8rMQ5LHqUaV-TU7Q1?usp=sharing)
- Training/evaluation plots: `results/plots/` (training reward, training success
  rate, epsilon decay, training loss, evaluation success by goal)

## Where to Find Things

| Looking for... | Go to |
|---|---|
| Quick results summary | This section, above |
| Full technical report | `report/DQN_Assignment_Report.pdf` |
| One-page Brightspace summary | `report/brightspace_summary.html` |
| Demonstration video | Google Drive link, above |
| All required plots | `results/plots/` |
| Trained model (selected) | `models/selected_dqn.pt` (a copy of `config_b_dqn.pt` -- Config B won the comparison) |
| Trained models (both configs) | `models/config_a_dqn.pt`, `models/config_b_dqn.pt` |
| Training metrics (both configs) | `results/config_a/training_metrics.csv`, `results/config_b/training_metrics.csv` |
| Evaluation results (DQN + rule-based) | `results/config_a_dqn_evaluation.csv`, `results/config_b_dqn_evaluation.csv`, `results/rule_based_evaluation.csv` |
| Config A vs. B comparison table | `results/plots/config_comparison.csv` |
| Runnable notebook (full workflow, end to end) | `DQN_Assignment3.ipynb` |
| Student-written DQN source code | `src/dqn/` |
| Gymnasium environment definition | `src/g1_rl/g1_elbow_env.py` |

## Assignment Deliverables Map

Mirrors the deliverables table in Section 10 of the assignment brief, in the
same order, so each required item can be checked directly against this repo.

| Deliverable | Required content | Where in this repo |
|---|---|---|
| Python source code | Complete student-written DQN implementation and any approved starter files that were modified | `src/dqn/` (student-written), `src/g1_rl/g1_elbow_env.py` (approved, unmodified) |
| Trained model | Saved PyTorch checkpoint for the selected configuration | `models/selected_dqn.pt` |
| Metrics files | CSV or equivalent structured output for both parameter configurations and final evaluation | `results/config_a/training_metrics.csv`, `results/config_b/training_metrics.csv`, `results/config_a_dqn_evaluation.csv`, `results/config_b_dqn_evaluation.csv`, `results/rule_based_evaluation.csv` |
| Plots | All required training, exploration, loss, success, and evaluation visualizations | `results/plots/` (5 required plots + `config_comparison.csv`) |
| Technical report | A concise academic report, recommended length 6-10 pages excluding appendices | `report/DQN_Assignment_Report.pdf` |
| Rendered video | A short 2-3 minute demonstration of the selected trained policy after headless training | See "Results at a Glance" above (Google Drive link) |
| README update | Exact commands for training, evaluation, checkpoint loading, and rendering | This file -- see "DQN Assignment" section below |

## Submission Checklist

Self-assessment against Section 14 of the assignment brief, at the time of
this submission:

- [x] Environment checker passes (`src/test_g1_elbow_env.py`).
- [x] Rule-based baseline results are recorded (`results/rule_based_evaluation.csv`).
- [x] Student-written DQN components are present (`src/dqn/`).
- [x] Both epsilon-decay experiments are complete (Config A and Config B, 1200 episodes each).
- [x] Training time is within five hours (each configuration trains in a few minutes on CPU).
- [x] CPU execution is supported (`torch.device("cpu")` throughout; CUDA optional and unused).
- [x] Selected checkpoint loads successfully (`models/selected_dqn.pt`).
- [x] Evaluation uses epsilon = 0.0 (`evaluate_dqn.py`, greedy).
- [x] Twenty evaluation episodes are reported (4 benchmark goals x 5 episodes).
- [x] Overall success rate is calculated correctly (100%, 20/20, both the DQN and the rule-based baseline).
- [x] Rule-based and DQN policies are compared (see "Results at a Glance" above).
- [x] All required plots are included (`results/plots/`).
- [x] The rendered video shows the trained DQN (epsilon = 0.0, loads `selected_dqn.pt`, does not retrain, two target angles).
- [x] README commands have been tested (executed end to end in `DQN_Assignment3.ipynb`).
- [x] The technical report and all required files are included (`report/DQN_Assignment_Report.pdf`).

---

## Project Summary

This project trains a student-written PyTorch DQN agent to control the
Unitree G1 left elbow across multiple target angles in a fixed-base MuJoCo
Gymnasium environment, compares it against a hand-written rule-based
baseline, evaluates success rate over 20 benchmark episodes, and
demonstrates the learned policy in the MuJoCo viewer. The DQN work is built
on top of the Unitree MuJoCo G1 Primer Workshop (background section near the
end of this file), which introduces control of a Unitree G1 humanoid robot
using MuJoCo and Gymnasium: model inspection, fixed-base model generation,
single-joint PD control, bias-force compensation, CSV logging, and a custom
Gymnasium environment with deterministic rule-based validation.

## Running the Notebook

The completed workflow notebook is `DQN_Assignment3.ipynb` at the
repository root. It imports the student-written source under `src/` (it
does not duplicate any logic) and runs the environment check, DQN
component overview, smoke test, training of both required configurations,
evaluation, plotting, and the rule-based comparison end to end.

```bash
source .venv/bin/activate
jupyter notebook DQN_Assignment3.ipynb
```

Run all cells top to bottom. Training both configurations takes a few
minutes on CPU; everything else (evaluation, plotting, table generation)
runs in seconds. The MuJoCo viewer demonstration (Section 12 of the
notebook) requires WSLg and is run separately from the command line, since
it opens a live display window that cannot run inside a notebook cell.

## Repository Structure

| Path | Contents |
|---|---|
| `DQN_Assignment3.ipynb` | Completed notebook: runs the full DQN workflow end to end (Section 16 deliverable). |
| `Unitree_MuJoCo_G1_Primer_Workshop.ipynb` | Primer workshop notebook (environment/model/PD-control stages, pre-DQN). |
| `src/` | All student-written Python source -- primer scripts, `g1_rl/` (Gymnasium environment), `dqn/` (DQN components), and the training/evaluation/plotting/rendering scripts described below. |
| `assets/` | The course-owned fixed-base G1 MuJoCo model and scene files. |
| `external/unitree_mujoco/` | Unmodified upstream Unitree MuJoCo repository (not edited, not committed -- see Setup). |
| `models/` | Saved PyTorch checkpoints: `config_a_dqn.pt`, `config_b_dqn.pt`, `selected_dqn.pt`. |
| `results/` | Training-metrics CSVs (`config_a/`, `config_b/`), evaluation CSVs, and `plots/` (all required figures and the config-comparison table). |
| `report/DQN_Assignment_Report.pdf` | Full technical report (Section 11 of the assignment). |
| `report/brightspace_summary.html` | Source for the one-page Brightspace submission PDF (Section 16.6). |
| `requirements.txt` / `requirements-lock.txt` | Top-level and fully pinned dependency lists. |
| `.gitignore` | Excludes virtual environments, caches, and other non-source artifacts. |

## Requirements

- Windows 11 with WSL 2
- Ubuntu 24.04
- Python 3.12
- WSLg for optional graphical demonstrations

## Setup

Run these commands from the repository root in WSL Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The workshop also uses the official Unitree MuJoCo repository as an external
dependency:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
git -C external/unitree_mujoco checkout ae6a8403e272733e9996ef59990880330496177f
```

## Workshop

Open `Unitree_MuJoCo_G1_Primer_Workshop.ipynb` and follow its sections in
order. Runtime source is under `src/`, and the course-owned fixed-base model is
under `assets/g1_fixed_base/`.

## Headless validation

```bash
source .venv/bin/activate
python -m compileall src
python src/inspect_g1_model.py \
  assets/g1_fixed_base/scene_29dof_fixed_base.xml \
  --no-viewer
python src/control_single_joint.py \
  --scene assets/g1_fixed_base/scene_29dof_fixed_base.xml \
  --target -0.8 \
  --duration 2 \
  --no-viewer
PYTHONPATH=src python src/test_g1_elbow_env.py
```

Headless execution is authoritative. Rendered demonstrations are optional and
require WSLg.

---

## DQN Assignment

### Environment, Actions, Reward, and Termination

**Environment:** `G1ElbowTargetEnv` (`src/g1_rl/g1_elbow_env.py`), approved
and unmodified from the primer workshop.

**Observation** (4 values): `[elbow_angle, elbow_velocity, goal_angle, goal_angle - elbow_angle]`

**Actions** (discrete, 3): `0` decrease the internal PD-controller target by
0.08 rad, `1` hold it, `2` increase it by 0.08 rad. The agent never commands
torque directly -- a PD controller plus MuJoCo `qfrc_bias` compensation
converts the target into bounded actuator torque.

**Reward:** `-|angle_error|` per step, `+1.0` on entering the 0.04 rad
success tolerance, `-0.05` for changing the target unnecessarily once
already inside that tolerance, `+10.0` terminal bonus on success.

**Success / termination:** the episode **terminates** when the elbow stays
within 0.04 rad of the goal for 8 consecutive steps (`info["is_success"]`).
It is **truncated** at 150 steps without meeting that streak. This
distinction matters for the Bellman target: `terminated` (not truncated) is
the only signal that zeroes bootstrapping, since a truncated episode ran out
of time while the task was still ongoing, not because it reached a true
terminal state.

### Required Hyperparameters

Baseline values required by Section 5.3 of the assignment, held identical
across both configurations except epsilon decay:

| Hyperparameter | Value |
|---|---|
| Discount factor (gamma) | 0.95 |
| Learning rate | 0.001 |
| Mini-batch size | 64 |
| Replay-buffer capacity | 50,000 transitions |
| Initial epsilon | 1.00 |
| Minimum epsilon | 0.05 |
| Epsilon decay -- Config A / Config B | 0.995 / 0.985 |
| Target-network update | Every 250 optimization steps |
| Warm-up before learning | At least 500 transitions |
| Maximum episode length | 150 steps |
| Training goal range | [-0.8, +0.8] rad |
| Evaluation epsilon | 0.00 |
| Training seed | 42 (Python, NumPy, PyTorch, environment) |
| Evaluation seed | 123 (action-selection RNG; evaluation is deterministic since epsilon = 0) |

**Note on the evaluation CSVs:** with epsilon = 0, fixed trained weights, and
deterministic MuJoCo physics, there is no remaining source of randomness
during evaluation. As a result, the 5 repeated episodes for each of the 4
benchmark goals in `results/config_a_dqn_evaluation.csv`,
`results/config_b_dqn_evaluation.csv`, and `results/rule_based_evaluation.csv`
report identical reward, episode length, and final error within each goal --
this is the expected outcome of a fully deterministic evaluation protocol,
not a data-generation error.

Student-written Deep Q-Network components live under `src/dqn/`:

| File | Contents |
|---|---|
| `src/dqn/q_network.py` | `QNetwork`: 4-in, 64-64 ReLU hidden, 3-out MLP. |
| `src/dqn/replay_buffer.py` | `ReplayBuffer`: fixed-capacity experience replay, stores `terminated` (not a combined `done`) so truncated episodes still bootstrap. |
| `src/dqn/agent.py` | `DQNAgent`: online/target networks, epsilon-greedy `select_action`, `optimize_model` (Bellman target, Huber loss, gradient clipping, target sync), checkpoint save/load. |
| `src/dqn/utils.py` | `set_global_seed`: seeds Python, NumPy, and PyTorch RNGs. |
| `src/train_dqn.py` | Headless training loop with CSV metrics logging. |
| `src/evaluate_dqn.py` | Greedy (`epsilon=0.0`) evaluation over the four required benchmark goals, 5 episodes each. |
| `src/evaluate_rule_based.py` | Evaluates `choose_rule_based_action()` on the identical benchmark protocol, for a fair comparison. |
| `src/render_dqn_policy.py` | Loads a saved checkpoint and demonstrates it in the MuJoCo viewer (does not train). |
| `src/generate_plots.py` | Produces all required training/evaluation plots and the config comparison table. |

### Reproducing the results

All commands below run from the repository root in WSL Ubuntu with the
virtual environment active:

```bash
source .venv/bin/activate
```

**1. Smoke test** (replay insertion, batch sampling, action selection, one
optimization step, checkpoint round-trip):

```bash
cd src
python -c "
from pathlib import Path
import numpy as np
from dqn import DQNAgent, ReplayBuffer, set_global_seed
from g1_rl import G1ElbowTargetEnv
set_global_seed(0)
rng = np.random.default_rng(0)
env = G1ElbowTargetEnv(goal_range=(-0.8, 0.8))
agent = DQNAgent(observation_dim=4, action_dim=3)
buffer = ReplayBuffer(1000)
obs, info = env.reset(seed=0)
for _ in range(80):
    action = agent.select_action(obs, epsilon=1.0, rng=rng)
    next_obs, reward, terminated, truncated, info = env.step(action)
    buffer.push(obs, action, reward, next_obs, terminated)
    obs = next_obs
    if terminated or truncated:
        obs, info = env.reset()
print('loss:', agent.optimize_model(buffer, batch_size=64))
env.close()
"
cd ..
```

**2. Train both required configurations** (headless, seed 42):

```bash
cd src
python train_dqn.py --config-name config_a --seed 42 --epsilon-decay 0.995 --max-episodes 1200 --max-minutes 45
python train_dqn.py --config-name config_b --seed 42 --epsilon-decay 0.985 --max-episodes 1200 --max-minutes 45
cd ..
```

Each run writes `results/<config-name>/training_metrics.csv` and
`models/<config-name>_dqn.pt`.

**3. Evaluate both configurations and the rule-based baseline** (greedy,
epsilon = 0.0, 20 benchmark episodes: goals -0.8/-0.4/+0.4/+0.8 rad, 5
episodes each):

```bash
cd src
python evaluate_dqn.py --checkpoint ../models/config_a_dqn.pt --seed 123
python evaluate_dqn.py --checkpoint ../models/config_b_dqn.pt --seed 123
python evaluate_rule_based.py
cd ..
```

**4. Select the stronger configuration and generate plots.** Config B
(faster epsilon decay, 0.985) was selected: both configurations reach 100%
benchmark success, but Config B reaches a 90%+ rolling training success rate
by episode 59 versus episode 80 for Config A, with slightly lower reward
variance in the final 50 training episodes. Its checkpoint is copied to
`models/selected_dqn.pt`.

```bash
cp models/config_b_dqn.pt models/selected_dqn.pt
cd src
python generate_plots.py \
  --selected-config-eval-csv ../results/config_b_dqn_evaluation.csv \
  --selected-config-label "Selected DQN (Config B)"
cd ..
```

Plots are written to `results/plots/`.

**5. Load the selected checkpoint and render the demonstration video**
(requires WSLg; loads the saved model, does not retrain):

```bash
cd src
python render_dqn_policy.py --checkpoint ../models/selected_dqn.pt --goals -0.8 0.8
cd ..
```

---

## Primer Workshop Background

This project develops a reproducible instructional workflow for working with the Unitree G1 humanoid robot in MuJoCo. The workshop portion (below) intentionally stops before implementation of a DQN agent; the "DQN Assignment" section above is where that agent is implemented, trained, and evaluated.

### Project Overview

The workshop begins with environment preparation and model inspection, then progresses through:

1. MuJoCo installation and viewer validation
2. Unitree G1 model inspection
3. Joint, actuator, sensor, `qpos`, and `qvel` analysis
4. Single-joint proportional-derivative control
5. Whole-body joint stabilization
6. Gravity and bias-force compensation
7. Creation of a course-owned fixed-base G1 model
8. CSV logging and deterministic validation
9. Construction of a custom Gymnasium environment
10. Rule-based environment validation
11. Optional interactive visualization before reinforcement learning

### Educational Purpose

The workshop is intended for college-level students studying:

- Reinforcement learning
- Robotics
- Machine learning
- Simulation
- Control systems
- Artificial intelligence
- Python programming

The material emphasizes conceptual understanding and reproducibility rather than only presenting finished code.

Students are expected to understand the relationship between:

```text
High-level discrete action
        ↓
Internal joint-position target
        ↓
PD controller
        ↓
Bias-force compensation
        ↓
Actuator torque
        ↓
Simulated physical movement
```

The workshop separates conventional low-level control from high-level reinforcement-learning decisions.

This allows students to focus on the reinforcement-learning problem without first needing to solve full humanoid balance, locomotion, inverse kinematics, and whole-body torque control.

### Learning Outcomes

After completing the workshop, students should be able to:

1. Explain the role of MuJoCo in robot simulation.
2. Distinguish between bodies, joints, actuators, sensors, and degrees of freedom.
3. Explain the purpose of `qpos` and `qvel`.
4. Load and inspect the Unitree G1 29-DOF model.
5. Identify a joint and actuator by name.
6. Read joint position and velocity data.
7. Apply bounded actuator torque.
8. Implement a proportional-derivative controller.
9. Explain the effect of gravity and bias forces.
10. Create a fixed-base instructional robot model.
11. Record simulation results in CSV format.
12. Build a Gymnasium-compatible environment.
13. Explain the difference between `terminated` and `truncated`.
14. Define observations, actions, rewards, and success conditions.
15. Validate an environment with a rule-based policy.
16. Confirm deterministic simulation behaviour.
17. Prepare the environment for a future student-written DQN agent.

### Current Project Status

| Milestone | Status |
|---|---|
| WSL 2 and Ubuntu setup | Complete |
| MuJoCo installation | Complete |
| MuJoCo viewer test | Complete |
| Unitree G1 repository integration | Complete |
| G1 model inspection | Complete |
| Fixed-base G1 generation | Complete |
| Left-elbow PD control | Complete |
| Whole-body joint stabilization | Complete |
| Bias-force compensation | Complete |
| CSV logging | Complete |
| Deterministic controller validation | Complete |
| Gymnasium environment | Complete |
| Gymnasium environment checker | Complete |
| Rule-based validation policy | Complete |
| Five-run determinism test | Complete |
| Optional rendered validation | Complete |
| Interactive camera-preparation demo | Complete |
| Student-written DQN | Complete (see DQN Assignment section above) |
| Physical G1 deployment | Future work |
