"""
Deep Q-Network (DQN) agent for Urban-Grid environment.
Uses temporal difference learning with experience replay and target networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Dict, Tuple, Optional


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture for urban planning.
    Processes grid-based observations and outputs Q-values for each action.
    """

    def __init__(self, grid_size: int, num_tile_types: int, hidden_dim: int = 256):
        super(DQNetwork, self).__init__()

        self.grid_size = grid_size
        self.num_tile_types = num_tile_types

        # Convolutional layers for grid processing with larger receptive field
        # Input: 5 channels (tile_grid, pop_g_grid, poll_g_grid, block_id_grid, road_connectivity_grid)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(5, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        conv_output_size = 128 * grid_size * grid_size

        self.feature_mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.combined_mlp = nn.Sequential(
            nn.Linear(conv_output_size + 256, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # Output: Q-value for each (row, col, tile_type) action
        # We'll output a separate Q-value head for each position and tile type
        self.q_value_head = nn.Linear(hidden_dim, grid_size * grid_size * num_tile_types)

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Dictionary with keys:
                - tile_grid: (batch, grid_size, grid_size)
                - pop_g_grid: (batch, grid_size, grid_size)
                - poll_g_grid: (batch, grid_size, grid_size)
                - block_id_grid: (batch, grid_size, grid_size)
                - road_connectivity_grid: (batch, grid_size, grid_size)
                - features: (batch, 4)

        Returns:
            Q-values: (batch, grid_size * grid_size * num_tile_types)
        """
        # Stack grids into channels
        grid_input = torch.stack([
            state['tile_grid'].float(),
            state['pop_g_grid'],
            state['poll_g_grid'],
            state['block_id_grid'].float(),
            state['road_connectivity_grid'].float()
        ], dim=1)  # (batch, 5, grid_size, grid_size)

        # Process grids through conv layers
        conv_features = self.conv_layers(grid_input)
        conv_features = conv_features.flatten(start_dim=1)

        # Process scalar features
        scalar_features = self.feature_mlp(state['features'])

        # Combine features
        combined = torch.cat([conv_features, scalar_features], dim=1)
        combined_features = self.combined_mlp(combined)

        # Output Q-values
        q_values = self.q_value_head(combined_features)

        return q_values


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Dict, action: np.ndarray, reward: float, next_state: Dict, done: bool):
        """Store a transition."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)

        # Separate batch components
        states = {
            'tile_grid': torch.stack([torch.tensor(e.state['tile_grid']) for e in experiences]),
            'pop_g_grid': torch.stack([torch.tensor(e.state['pop_g_grid']) for e in experiences]),
            'poll_g_grid': torch.stack([torch.tensor(e.state['poll_g_grid']) for e in experiences]),
            'block_id_grid': torch.stack([torch.tensor(e.state['block_id_grid']) for e in experiences]),
            'road_connectivity_grid': torch.stack([torch.tensor(e.state['road_connectivity_grid']) for e in experiences]),
            'features': torch.stack([torch.tensor(e.state['features']) for e in experiences])
        }

        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)

        next_states = {
            'tile_grid': torch.stack([torch.tensor(e.next_state['tile_grid']) for e in experiences]),
            'pop_g_grid': torch.stack([torch.tensor(e.next_state['pop_g_grid']) for e in experiences]),
            'poll_g_grid': torch.stack([torch.tensor(e.next_state['poll_g_grid']) for e in experiences]),
            'block_id_grid': torch.stack([torch.tensor(e.next_state['block_id_grid']) for e in experiences]),
            'road_connectivity_grid': torch.stack([torch.tensor(e.next_state['road_connectivity_grid']) for e in experiences]),
            'features': torch.stack([torch.tensor(e.next_state['features']) for e in experiences])
        }

        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent using temporal difference learning.

    Features:
    - Experience replay for stable learning
    - Target network for stable Q-value estimation
    - Epsilon-greedy exploration
    - Double DQN option
    """

    def __init__(
        self,
        grid_size: int = 16,
        num_tile_types: int = 5,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dim: int = 256,
        device: str = 'cpu',
        double_dqn: bool = True
    ):
        self.grid_size = grid_size
        self.num_tile_types = num_tile_types
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.device = torch.device(device)

        # Q-networks
        self.policy_net = DQNetwork(grid_size, num_tile_types, hidden_dim).to(self.device)
        self.target_net = DQNetwork(grid_size, num_tile_types, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.steps = 0

    def select_action(self, state: Dict[str, np.ndarray], valid_actions: Optional[np.ndarray] = None, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation
            valid_actions: Optional mask of valid actions (barren cells)
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: [row, col, tile_type]
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Random action
            if valid_actions is not None and len(valid_actions) > 0:
                return valid_actions[random.randint(0, len(valid_actions) - 1)]
            else:
                row = random.randint(0, self.grid_size - 1)
                col = random.randint(0, self.grid_size - 1)
                tile_type = random.randint(0, self.num_tile_types - 1)
                return np.array([row, col, tile_type])

        # Greedy action
        with torch.no_grad():
            state_tensor = self._dict_to_tensor(state)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

            # Reshape to (grid_size, grid_size, num_tile_types)
            q_values = q_values.reshape(self.grid_size, self.grid_size, self.num_tile_types)

            # If valid actions provided, mask invalid actions
            if valid_actions is not None and len(valid_actions) > 0:
                # Find action with highest Q-value among valid actions
                best_action_idx = None
                best_q_value = -np.inf
                for action in valid_actions:
                    row, col, tile_type = action
                    q = q_values[row, col, tile_type]
                    if q > best_q_value:
                        best_q_value = q
                        best_action_idx = action
                return best_action_idx
            else:
                # Select action with highest Q-value
                best_action = np.unravel_index(q_values.argmax(), q_values.shape)
                return np.array(best_action)

    def train_step(self):
        """Perform one training step using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Move to device
        states = {k: v.to(self.device) for k, v in states.items()}
        next_states = {k: v.to(self.device) for k, v in next_states.items()}
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(states)

        # Convert actions to flat indices
        # action = [row, col, tile_type] -> flat_idx = row * (grid_size * num_tile_types) + col * num_tile_types + tile_type
        flat_actions = (
            actions[:, 0] * (self.grid_size * self.num_tile_types) +
            actions[:, 1] * self.num_tile_types +
            actions[:, 2]
        )

        current_q_values = current_q_values.gather(1, flat_actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_q_values_policy = self.policy_net(next_states)
                next_actions = next_q_values_policy.argmax(dim=1)
                next_q_values_target = self.target_net(next_states)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(dim=1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Huber loss is more stable than MSE)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def store_experience(self, state: Dict, action: np.ndarray, reward: float, next_state: Dict, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _dict_to_tensor(self, state: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy state dict to tensor dict with batch dimension."""
        return {
            k: torch.tensor(v).unsqueeze(0).to(self.device)
            for k, v in state.items()
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
