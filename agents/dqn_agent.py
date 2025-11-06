# agents/dqn_agent.py
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _to_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        arr = np.array(x)
    elif np.isscalar(x):
        arr = np.array([x], dtype=np.float32)
    else:
        raise TypeError(f"Unsupported observation type: {type(x)}")
    return arr.astype(np.float32).reshape(-1)


def _choose_from_valid(qvals: torch.Tensor, valid_idx: Optional[np.ndarray]) -> int:
    if valid_idx is None:
        return int(torch.argmax(qvals).item())
    full = torch.full_like(qvals, float("-inf"))
    full[valid_idx] = qvals[valid_idx]
    return int(torch.argmax(full).item())


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    start_learning_after: int = 1_000
    train_freq: int = 1
    target_update_freq: int = 1_000
    tau: float = 1.0  # 1.0 = hard copy, <1 soft
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.s = np.zeros((capacity, 1), dtype=object)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.ns = np.zeros((capacity, 1), dtype=object)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def push(self, s, a, r, ns, d):
        i = self.ptr
        self.s[i, 0] = _to_np(s)
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.ns[i, 0] = _to_np(ns)
        self.d[i] = bool(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        s = np.stack(self.s[idx, 0], axis=0)
        ns = np.stack(self.ns[idx, 0], axis=0)
        return s.astype(np.float32), self.a[idx], self.r[idx], ns.astype(np.float32), self.d[idx]

    def __len__(self):
        return self.size


class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        cfg: DQNConfig = DQNConfig(),
        seed: int = 42,
        save_dir: str = "data/outputs/dqn",
        device: Optional[str] = None,
    ):
        self.obs_size = int(obs_size)
        self.n_actions = int(n_actions)
        self.cfg = cfg
        self.save_dir = save_dir

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.q = QNet(self.obs_size, self.n_actions).to(self.device)
        self.target = QNet(self.obs_size, self.n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=self.cfg.lr)

        self.rb = ReplayBuffer(self.cfg.replay_capacity)
        self.global_step = 0

        self.eps_start = self.cfg.eps_start
        self.eps_end = self.cfg.eps_end
        self.eps_decay = max(1, int(self.cfg.eps_decay_steps))

        os.makedirs(self.save_dir, exist_ok=True)

    def reset(self):
        pass

    def _epsilon(self) -> float:
        t = min(self.global_step, self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - t / self.eps_decay)

    @torch.no_grad()
    def act(self, obs, valid_actions: Optional[np.ndarray] = None) -> int:
        eps = self._epsilon()
        self.global_step += 1

        if np.random.rand() < eps:
            if valid_actions is not None and len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return int(np.random.randint(0, self.n_actions))

        x = torch.from_numpy(_to_np(obs)).to(self.device).unsqueeze(0)
        qvals = self.q(x).squeeze(0)
        return _choose_from_valid(qvals, valid_actions)

    def learn(self, transition):
        s, a, r, ns, d = transition
        self.rb.push(s, a, r, ns, d)

        if self.global_step < self.cfg.start_learning_after:
            return
        if self.global_step % self.cfg.train_freq != 0:
            return
        if len(self.rb) < self.cfg.batch_size:
            return

        b_s, b_a, b_r, b_ns, b_d = self.rb.sample(self.cfg.batch_size)
        s_t = torch.from_numpy(b_s).to(self.device)
        ns_t = torch.from_numpy(b_ns).to(self.device)
        a_t = torch.from_numpy(b_a).to(self.device).long()
        r_t = torch.from_numpy(b_r).to(self.device)
        d_t = torch.from_numpy(b_d.astype(np.float32)).to(self.device)

        q_pred = self.q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.q(ns_t), dim=1)
            next_q = self.target(ns_t).gather(1, next_actions.view(-1, 1)).squeeze(1)
            target = r_t + self.cfg.gamma * (1.0 - d_t) * next_q

        loss = nn.functional.smooth_l1_loss(q_pred, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.optim.step()

        if self.cfg.tau >= 1.0:
            if self.global_step % self.cfg.target_update_freq == 0:
                self.target.load_state_dict(self.q.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.target.parameters()):
                    tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def save(self, tag: str = "final") -> str:
        path = os.path.join(self.save_dir, f"dqn_{tag}.pt")
        torch.save(
            {
                "q_state_dict": self.q.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "cfg": self.cfg.__dict__,
                "obs_size": self.obs_size,
                "n_actions": self.n_actions,
                "global_step": self.global_step,
            },
            path,
        )
        return path

    def load(self, path: str):
        chk = torch.load(path, map_location=self.device)
        self.q.load_state_dict(chk["q_state_dict"])
        self.target.load_state_dict(chk.get("target_state_dict", chk["q_state_dict"]))
        self.obs_size = chk.get("obs_size", self.obs_size)
        self.n_actions = chk.get("n_actions", self.n_actions)
        self.global_step = chk.get("global_step", 0)
