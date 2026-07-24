# DQN Control of the Unitree G1 Left Elbow

**Assignment:** CSCN8020 DQN Assignment -- Deep Q-Network Control of the Unitree G1 Left Elbow
**Student:** Liggia Cruz
**Student ID:** 9085905
**Instructor:** Prof. Enrique Espinosa, Conestoga College
**Repository:** https://github.com/liggiaelena/CSCN8020_Assignment3
**Clone URL:** https://github.com/liggiaelena/CSCN8020_Assignment3.git
**Validated run environment:** Python 3.12, WSL2 Ubuntu 24.04 (exact package versions pinned in `requirements-lock.txt`)


## Project Summary

This project trains a student-written PyTorch DQN agent to control the
Unitree G1 left elbow across multiple target angles in a fixed-base MuJoCo
Gymnasium environment, compares it against a hand-written rule-based
baseline, evaluates success rate over 20 benchmark episodes, and
demonstrates the learned policy in the MuJoCo viewer. The DQN work is built
on top of the Unitree MuJoCo G1 Primer Workshop below, which introduces
control of a Unitree G1 humanoid robot using MuJoCo and Gymnasium: model
inspection, fixed-base model generation, single-joint PD control,
bias-force compensation, CSV logging, and a custom Gymnasium environment
with deterministic rule-based validation.

The workshop portion intentionally stops before implementation of a DQN
agent; the "DQN Assignment" section further below is where that agent is
implemented, trained, and evaluated.

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

## Project Overview

This project develops a reproducible instructional workflow for working with the Unitree G1 humanoid robot in MuJoCo.

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

The project intentionally stops before the student-written Deep Q-Network implementation. The validated Gymnasium environment is designed to become the foundation for that next phase.

---

## Educational Purpose

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

---

## Learning Outcomes

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

---

## Current Project Status

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
| Student-written DQN | Complete (see DQN Assignment section below) |
| Physical G1 deployment | Future work |

---

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
