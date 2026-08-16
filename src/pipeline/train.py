"""
Unified Training Pipeline for Contextual Reinforcement Learning Benchmarking.
Supports training across benchmark algorithms:
  - context_free: PPO on state observation only
  - oracle: PPO on state observation + ground-truth context parameters
  - vae: PPO augmented with Variational Autoencoder (VAE) representation
  - cpc: PPO augmented with Contrastive Predictive Coding (CPC) representation
"""
from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.base_ppo import BasePPOAgent
from src.models.vae_encoder import VAEContextEncoder
from src.models.cpc_encoder import CPCContextEncoder
from src.utils.env_utils import (
    make_env,
    sample_contexts,
    extract_input,
    extract_raw_state,
    resolve_path,
)
from src.utils.experiment_logger import ExperimentLogger
from src.utils.history_buffer import HistoryBuffer
from src.utils.plotting import Plotter


def _get_context_id(env, obs=None) -> int:
    ctx = None
    if isinstance(obs, dict) and "context" in obs:
        ctx = obs["context"]
    elif hasattr(env, "context"):
        ctx = getattr(env, "context")
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "context"):
        ctx = getattr(env.unwrapped, "context")

    if isinstance(ctx, dict):
        g = ctx.get("gravity", ctx.get("g", 0.0))
        return int(round(float(g) * 10))
    return 0


def train_single_run(
        mode: str = "context_free",
        env_name: str = "CARLPendulum",
        total_steps: int = 51200,
        rollout_steps: int = 2048,
        seed: int = 0,
        num_contexts: int = 100,
        train_context_seed: int = 42,
        vary_contexts: Optional[List[str]] = None,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        encoder_lr: float = 3e-4,
        kl_weight: float = 1e-3,
        cpc_temperature: float = 0.1,
        predict_horizon: int = 3,
        max_history_len: int = 25,
        exp_tag: str = "",
        output_dir: str = "results",
        use_bo_config: bool = True,
) -> str:
    """
    Trains a single RL model checkpoint on training context environments.
    """
    output_dir = resolve_path(output_dir)

    if use_bo_config:
        best_yaml_path = resolve_path(os.path.join("configs", f"best_hyperparams_{mode}.yaml"))
        if os.path.exists(best_yaml_path):
            try:
                with open(best_yaml_path, "r") as f:
                    bo_data = yaml.safe_load(f)
                if bo_data and "hyperparameters" in bo_data:
                    cfg_env = bo_data.get("env_name", None)
                    if cfg_env is None or cfg_env.lower() == env_name.lower():
                        hp = bo_data["hyperparameters"]
                        latent_dim = hp.get("latent_dim", latent_dim)
                        hidden_dim = hp.get("hidden_dim", hidden_dim)
                        encoder_lr = hp.get("encoder_lr", encoder_lr)
                        kl_weight = hp.get("kl_weight", kl_weight)
                        cpc_temperature = hp.get("cpc_temperature", cpc_temperature)
                        predict_horizon = hp.get("predict_horizon", predict_horizon)
                        max_history_len = hp.get("max_history_len", max_history_len)
                        print(
                            f"[BO CONFIG] Loaded tuned BO hyperparameters for '{mode.upper()}' ({env_name}) from '{best_yaml_path}'")
                    else:
                        print(
                            f"[BO CONFIG] Ignored BO hyperparams for '{mode.upper()}' because environment mismatch ({cfg_env} vs {env_name})")
            except Exception as e:
                print(f"[WARNING] Could not load BO hyperparameters from '{best_yaml_path}': {e}")

    torch.set_num_threads(1)

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(
        env_name,
        seed=seed,
        num_contexts=num_contexts,
        context_seed=train_context_seed,
        vary_contexts=vary_contexts,
    )

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    action_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    tag_str = f"_{exp_tag}" if exp_tag else ""
    folder_name = f"{total_steps}steps_{env_name.lower().replace('-', '_')}_{mode}_seed{seed}{tag_str}"
    run_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    exp_name = f"{mode}_seed{seed}"
    logger = ExperimentLogger(output_dir=run_dir, experiment_name=exp_name)

    obs, _ = env.reset(seed=seed)
    raw_state = extract_raw_state(obs)
    raw_state_dim = len(raw_state)

    history_buf: Optional[HistoryBuffer] = None
    vae_encoder: Optional[VAEContextEncoder] = None
    cpc_encoder: Optional[CPCContextEncoder] = None
    encoder_optimizer: Optional[torch.optim.Optimizer] = None
    device = torch.device("cpu")
    device_name = "CPU"

    if mode in ["context_free", "oracle"]:
        initial_input = extract_input(obs, env, mode)
        input_dim = len(initial_input)
    else:
        history_buf = HistoryBuffer(
            max_history_len=max_history_len,
            state_dim=raw_state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
        )
        input_dim = raw_state_dim + latent_dim

        if mode == "vae":
            vae_encoder = VAEContextEncoder(
                state_dim=raw_state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                predict_horizon=predict_horizon,
                kl_weight=kl_weight,
                device=device,
            )
            encoder_optimizer = torch.optim.Adam(vae_encoder.parameters(), lr=encoder_lr, eps=1e-5)
        elif mode == "cpc":
            cpc_encoder = CPCContextEncoder(
                state_dim=raw_state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                predict_horizon=predict_horizon,
                temperature=cpc_temperature,
                device=device,
            )
            encoder_optimizer = torch.optim.Adam(cpc_encoder.parameters(), lr=encoder_lr, eps=1e-5)

    agent = BasePPOAgent(input_dim=input_dim, action_dim=action_dim, is_continuous=is_continuous, device=device)

    hyperparams = {
        "env_name": env_name,
        "mode": mode,
        "seed": seed,
        "train_context_seed": train_context_seed,
        "num_contexts": num_contexts,
        "vary_contexts": vary_contexts,
        "total_steps": total_steps,
        "rollout_steps": rollout_steps,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "encoder_lr": encoder_lr,
        "kl_weight": kl_weight,
        "cpc_temperature": cpc_temperature,
        "predict_horizon": predict_horizon,
        "max_history_len": max_history_len,
        "is_continuous": is_continuous,
        "raw_state_dim": raw_state_dim,
        "input_dim": input_dim,
        "action_dim": action_dim,
    }
    logger.save_config(hyperparams)

    global_step = 0
    ep_return = 0.0
    completed_returns: List[float] = []

    print(
        f"\n[TRAIN] Mode: {mode.upper()} | Env: {env_name} | Seed: {seed} | Device: {device_name} | Input Dim: {input_dim}")

    curr_obs = obs
    curr_hist_tensor = history_buf.get_tensor(pad_to_max=True) if history_buf else None
    curr_z_np = None

    def _extract_z(h_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if mode == "vae" and vae_encoder is not None:
                mu_z, _ = vae_encoder.encode(h_tensor)
                return mu_z.squeeze(0).cpu().numpy()
            elif mode == "cpc" and cpc_encoder is not None:
                z_t = cpc_encoder.encode(h_tensor)
                return z_t.squeeze(0).cpu().numpy()
            return np.zeros((latent_dim,), dtype=np.float32)

    if mode in ["vae", "cpc"] and curr_hist_tensor is not None:
        curr_z_np = _extract_z(curr_hist_tensor)

    while global_step < total_steps:
        trajectory = []
        for _ in range(rollout_steps):
            context_id = _get_context_id(env, curr_obs)

            if mode in ["context_free", "oracle"]:
                policy_input = extract_input(curr_obs, env, mode)
                action, logp, ent, val = agent.predict(policy_input)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                next_policy_input = extract_input(next_obs, env, mode)
                trajectory.append(
                    (policy_input, action, logp, ent, reward, term, trunc, next_policy_input, None, None, None,
                     context_id))
            else:
                raw_state = extract_raw_state(curr_obs)
                policy_input = np.concatenate([raw_state, curr_z_np], axis=0)

                action, logp, ent, val = agent.predict(policy_input)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                history_buf.add(raw_state, action, float(reward))
                next_raw_state = extract_raw_state(next_obs)
                next_hist_tensor = history_buf.get_tensor(pad_to_max=True)
                next_z_np = _extract_z(next_hist_tensor)
                next_policy_input = np.concatenate([next_raw_state, next_z_np], axis=0)

                trajectory.append((
                    policy_input,
                    action,
                    logp,
                    ent,
                    reward,
                    term,
                    trunc,
                    next_policy_input,
                    curr_hist_tensor,
                    raw_state,
                    next_raw_state,
                    context_id,
                ))
                curr_hist_tensor = next_hist_tensor
                curr_z_np = next_z_np

            ep_return += float(reward)
            global_step += 1

            if done:
                completed_returns.append(ep_return)
                ep_return = 0.0
                if history_buf:
                    history_buf.reset()
                    curr_hist_tensor = history_buf.get_tensor(pad_to_max=True)
                    if mode in ["vae", "cpc"]:
                        curr_z_np = _extract_z(curr_hist_tensor)
                curr_obs, _ = env.reset()
            else:
                curr_obs = next_obs

        repr_loss_val = 0.0
        if mode in ["vae", "cpc"] and encoder_optimizer is not None:
            N = len(trajectory)
            K = max(1, predict_horizon)
            hist_list = []
            state_seq_list = []
            action_seq_list = []
            next_state_seq_list = []
            reward_seq_list = []
            valid_seq_list = []
            ep_ids = []

            for idx in range(N):
                hist_list.append(trajectory[idx][8])
                ep_ids.append(trajectory[idx][11])
                s_seq, a_seq, ns_seq, r_seq, valid_seq = [], [], [], [], []
                curr_i = idx
                is_valid = 1.0
                for k in range(K):
                    st = trajectory[curr_i]
                    s_seq.append(st[9])
                    raw_act = st[1]
                    if is_continuous:
                        a_seq.append(np.array(raw_act, dtype=np.float32))
                    else:
                        act_vec = np.zeros(action_dim, dtype=np.float32)
                        act_vec[int(raw_act)] = 1.0
                        a_seq.append(act_vec)
                    ns_seq.append(st[10])
                    r_seq.append([st[4]])
                    valid_seq.append([is_valid])

                    done = st[5] or st[6]
                    if done or curr_i + 1 >= N:
                        is_valid = 0.0
                    else:
                        curr_i += 1

                state_seq_list.append(s_seq)
                action_seq_list.append(a_seq)
                next_state_seq_list.append(ns_seq)
                reward_seq_list.append(r_seq)
                valid_seq_list.append(valid_seq)

            hist_batch = torch.cat(hist_list, dim=0)
            state_seq_batch = torch.tensor(np.array(state_seq_list), dtype=torch.float32)
            action_seq_batch = torch.tensor(np.array(action_seq_list), dtype=torch.float32)
            next_state_seq_batch = torch.tensor(np.array(next_state_seq_list), dtype=torch.float32)
            reward_seq_batch = torch.tensor(np.array(reward_seq_list), dtype=torch.float32)
            valid_seq_batch = torch.tensor(np.array(valid_seq_list), dtype=torch.float32)
            ep_ids = np.array(ep_ids)

            encoder_epochs = 1
            batch_size = 64
            num_samples = len(hist_batch)
            indices = np.arange(num_samples)
            active_encoder = vae_encoder if mode == "vae" else cpc_encoder

            for epoch in range(encoder_epochs):
                np.random.shuffle(indices)
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    mb_idx = indices[start_idx:end_idx]

                    mb_hist = hist_batch[mb_idx]
                    mb_state = state_seq_batch[mb_idx]
                    mb_action = action_seq_batch[mb_idx]
                    mb_next_state = next_state_seq_batch[mb_idx]
                    mb_reward = reward_seq_batch[mb_idx]
                    mb_valid_mask = valid_seq_batch[mb_idx].to(device)
                    mb_ep_ids = torch.tensor(ep_ids[mb_idx], dtype=torch.long, device=device)

                    encoder_optimizer.zero_grad()

                    if mode == "vae" and vae_encoder is not None:
                        loss_dict = vae_encoder.compute_loss(mb_hist, mb_state, mb_action, mb_next_state, mb_reward)
                    elif mode == "cpc" and cpc_encoder is not None:
                        loss_dict = cpc_encoder.compute_loss(
                            mb_hist,
                            mb_state,
                            mb_action,
                            mb_next_state,
                            mb_reward,
                            ep_ids=mb_ep_ids,
                            valid_mask=mb_valid_mask,
                        )

                    repr_loss = loss_dict["loss"]
                    repr_loss.backward()

                    if active_encoder is not None:
                        torch.nn.utils.clip_grad_norm_(active_encoder.parameters(), max_norm=1.0)

                    encoder_optimizer.step()
                    repr_loss_val = repr_loss.item()
                    last_loss_dict = {k: v.item() for k, v in loss_dict.items()}

        ppo_traj = [(s, a, lp, e, r, tm, tr, ns, h) for (s, a, lp, e, r, tm, tr, ns, h, *_) in trajectory]
        p_loss, v_loss, e_loss = agent.update(ppo_traj)

        recent_returns = completed_returns[-10:] if completed_returns else [0.0]
        mean_ret = float(np.mean(recent_returns))

        log_dict = {
            "step": global_step,
            "mean_return": mean_ret,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "entropy_loss": e_loss,
        }

        if mode == "vae":
            log_dict["vae_loss"] = repr_loss_val
        elif mode == "cpc":
            log_dict["cpc_loss"] = repr_loss_val
            if 'last_loss_dict' in locals():
                log_dict["infonce_loss"] = last_loss_dict.get("cpc_loss", 0.0)
                log_dict["supcon_loss"] = last_loss_dict.get("supcon_loss", 0.0)
                log_dict["aux_dyn_loss"] = last_loss_dict.get("aux_dyn_loss", 0.0)

        logger.log(log_dict)

        loss_label = "VAE" if mode == "vae" else "CPC"
        loss_str = ""

        if mode == "vae":
            loss_str = f"VAE Loss: {repr_loss_val:6.3f} | "
        elif mode == "cpc":
            if 'last_loss_dict' in locals():
                inf = last_loss_dict.get("cpc_loss", 0.0)
                sup = last_loss_dict.get("supcon_loss", 0.0)
                aux = last_loss_dict.get("aux_dyn_loss", 0.0)
                loss_str = f"CPC Total: {repr_loss_val:5.2f} [InfoNCE: {inf:4.2f} | SupCon (w): {sup * 0.05:4.2f} | AuxMSE (w): {aux * 2.0:4.2f}] | "
            else:
                loss_str = f"CPC Loss: {repr_loss_val:6.3f} | "

        print(
            f"Step {global_step:7d}/{total_steps:7d} | "
            f"Mean Return (last 10 eps): {mean_ret:6.1f} | "
            f"{loss_str}"
            f"Value Loss: {v_loss:7.2f}"
        )

    logger.save()
    agent.save_checkpoint(os.path.join(run_dir, "agent.pt"))
    if vae_encoder:
        torch.save(vae_encoder.state_dict(), os.path.join(run_dir, "vae_encoder.pt"))
    if cpc_encoder:
        torch.save(cpc_encoder.state_dict(), os.path.join(run_dir, "cpc_encoder.pt"))

    print(f"[SUCCESS] Saved trained model checkpoint to '{run_dir}'")
    return run_dir