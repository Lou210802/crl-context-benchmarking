"""
Trajectory History Buffer for Contextual Reinforcement Learning.

This module provides the HistoryBuffer class for storing sliding-window sequence histories
of states, actions, and rewards within an episode, enabling recurrent context encoding.
"""

from typing import List, Tuple, Any, Optional
import numpy as np
import torch


class HistoryBuffer:
    """
    Sliding window buffer storing recent (state, action, reward) transitions within an episode.
    """

    def __init__(self, max_history_len: int = 25, state_dim: int = 3, action_dim: int = 1, is_continuous: bool = True) -> None:
        """
        Initializes max history length and transition dimensions.

        Args:
            max_history_len: Maximum number of past steps to keep in sliding window.
            state_dim: Dimension of state observation vector.
            action_dim: Dimension of action vector.
            is_continuous: Whether the action space is continuous (True) or discrete (False).
        """
        self.max_history_len = max_history_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.reset()

    def reset(self) -> None:
        """
        Clears the buffer at episode start.
        """
        self.history: List[Tuple[np.ndarray, np.ndarray, float]] = []

    def add(self, state: np.ndarray, action: Any, reward: float) -> None:
        """
        Adds a single (state, action, reward) transition to the sliding history.

        Args:
            state: State observation vector.
            action: Action taken (int for discrete, np.ndarray for continuous).
            reward: Scalar reward received.
        """
        state_arr = np.array(state, dtype=np.float32).flatten()
        if self.is_continuous:
            act_arr = np.array(action, dtype=np.float32).flatten()
        else:
            # One-hot encode discrete action
            act_arr = np.zeros(self.action_dim, dtype=np.float32)
            act_arr[int(action)] = 1.0

        self.history.append((state_arr, act_arr, float(reward)))
        if len(self.history) > self.max_history_len:
            self.history.pop(0)

    def get_tensor(self) -> torch.Tensor:
        """
        Returns history transition sequence as a PyTorch tensor of shape (1, seq_len, feat_dim).
        If history is empty, returns a zero tensor of shape (1, 1, feat_dim).

        Returns:
            torch.Tensor: PyTorch tensor of shape (1, sequence_length, feature_dimension).
        """
        feat_dim = self.state_dim + self.action_dim + 1
        if len(self.history) == 0:
            return torch.zeros((1, 1, feat_dim), dtype=torch.float32)

        seq_feats = []
        for s, a, r in self.history:
            feat = np.concatenate([s, a, [r]], axis=0)
            seq_feats.append(feat)

        tensor = torch.tensor(np.array(seq_feats), dtype=torch.float32).unsqueeze(0)
        return tensor
