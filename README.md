# Contextual RL Representation Benchmarking

A benchmarking and evaluation framework for Contextual Reinforcement Learning (cRL) comparing context representation learning methods (VAE, CPC) against standard baselines (Context-Free and Oracle) on CARL environments (e.g., `CARLPendulum`).

## Installation

Prerequisites: Python >= 3.12

```bash
git clone https://github.com/Lou210802/crl-context-benchmarking.git
cd crl-context-benchmarking

# Using uv
uv sync

# Or using pip
pip install -e .
```

## Usage

All commands are executed via [`src/main.py`](file:///home/lou/PycharmProjects/crl-context-benchmarking/src/main.py).

### 1. End-to-End Benchmark (`run-all`)

Runs hyperparameter optimization (Bayesian Optimization), training across seeds, evaluation on held-out contexts, and plot generation:

```bash
python src/main.py run-all --env CARLPendulum --modes context_free oracle vae cpc --seeds 0 1 2 --total_steps 25000 --num_workers 4
```

Use `--skip_bo` to bypass Bayesian Optimization and use existing configs in `configs/best_hyperparams_<mode>.yaml`.

### 2. Training (`train`)

Train specific modes across seeds in parallel:

```bash
python src/main.py train --modes context_free oracle vae cpc --seeds 0 1 2 3 --total_steps 25000 --num_workers 4
```

Supported modes: `context_free`, `oracle`, `vae`, `cpc`.

### 3. Evaluation (`eval`)

Evaluate trained models on held-out physical contexts:

```bash
python src/main.py eval --dir results --eval_episodes 20 --num_eval_contexts 25
```

### 4. Hyperparameter Optimization (`optimize`)

Run Bayesian Optimization via Optuna for encoder or PPO hyperparameters:

```bash
python src/main.py optimize --modes vae cpc --n_trials 30 --bo_steps 25000
```

### 5. Step-Budget Pilot Runs (`pilot`)

Run step-budget convergence analysis:

```bash
python src/main.py pilot --modes context_free oracle vae cpc --step_candidates 5000 10000 15000 25000 40000
```

### 6. Plotting (`plot`)

Launch the interactive terminal plotting menu:

```bash
python src/main.py plot --dir results
```
