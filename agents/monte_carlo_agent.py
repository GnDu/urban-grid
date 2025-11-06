"""
Monte Carlo RL agent for Urban-Grid environment.
Uses episodic learning with first-visit or every-visit MC methods.
"""

import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class MonteCarloAgent:
    """
    Monte Carlo agent using episodic learning.

    Features:
    - First-visit or every-visit MC
    - Epsilon-greedy exploration
    - Simple state representation for tractability
    - Stores Q(s,a) values in a dictionary
    """

    def __init__(
        self,
        grid_size: int = 16,
        num_tile_types: int = 5,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        first_visit: bool = True
    ):
        self.grid_size = grid_size
        self.num_tile_types = num_tile_types
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.first_visit = first_visit

        # Q-values: Q(s, a) stored as dictionary
        self.q_values = defaultdict(lambda: defaultdict(float))

        # Returns for each state-action pair
        self.returns = defaultdict(lambda: defaultdict(list))

        # Statistics
        self.episodes_trained = 0

    def _state_to_key(self, state: Dict[str, np.ndarray]) -> str:
        """
        Convert state to hashable key for Q-table.
        Uses simplified state representation for tractability.

        State features:
        - Tile type counts (6 values)
        - Total population/pollution (2 values)
        - Number of barren cells (1 value)
        """
        tile_grid = state['tile_grid']
        features = state['features']

        # Count each tile type
        tile_counts = []
        for tile_type in range(6):
            count = np.sum(tile_grid == tile_type)
            tile_counts.append(count)

        # Extract scalar features
        time_step = int(features[0])
        total_pop = float(features[1])
        total_poll = float(features[2])
        pop_cap = float(features[3])

        # Create simple state representation
        # Discretize continuous values to reduce state space
        total_pop_bin = int(total_pop // 10)
        total_poll_bin = int(total_poll // 10)
        pop_cap_bin = int(pop_cap // 10)

        state_key = (
            tuple(tile_counts),
            time_step,
            total_pop_bin,
            total_poll_bin,
            pop_cap_bin
        )

        return str(state_key)

    def _action_to_key(self, action: np.ndarray) -> str:
        """Convert action to hashable key."""
        return f"{action[0]}_{action[1]}_{action[2]}"

    def select_action(
        self,
        state: Dict[str, np.ndarray],
        valid_actions: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation
            valid_actions: List of valid actions
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: [row, col, tile_type]
        """
        state_key = self._state_to_key(state)

        # Get valid actions
        if valid_actions is None or len(valid_actions) == 0:
            # Generate all possible actions
            valid_actions = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    for tile_type in range(self.num_tile_types):
                        valid_actions.append([row, col, tile_type])
            valid_actions = np.array(valid_actions)

        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            # Random action
            return valid_actions[np.random.randint(0, len(valid_actions))]

        # Greedy action: select action with highest Q-value
        best_action = None
        best_q_value = -np.inf

        for action in valid_actions:
            action_key = self._action_to_key(action)
            q_value = self.q_values[state_key][action_key]

            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        # If no Q-values exist yet, random action
        if best_action is None:
            best_action = valid_actions[np.random.randint(0, len(valid_actions))]

        return best_action

    def train_episode(self, episode: List[Tuple[Dict, np.ndarray, float]]):
        """
        Train on a complete episode using Monte Carlo learning.

        Args:
            episode: List of (state, action, reward) tuples
        """
        # Calculate returns (G) for each timestep
        G = 0
        visited = set()

        # Iterate backwards through episode
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            state_key = self._state_to_key(state)
            action_key = self._action_to_key(action)

            # Update return
            G = reward + self.gamma * G

            # Create state-action pair key
            sa_pair = (state_key, action_key)

            # First-visit MC: only update if this is first visit to (s,a)
            if self.first_visit and sa_pair in visited:
                continue

            visited.add(sa_pair)

            # Store return
            self.returns[state_key][action_key].append(G)

            # Update Q-value as average of returns
            self.q_values[state_key][action_key] = np.mean(
                self.returns[state_key][action_key]
            )

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.episodes_trained += 1

    def get_q_value(self, state: Dict[str, np.ndarray], action: np.ndarray) -> float:
        """Get Q-value for a state-action pair."""
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        return self.q_values[state_key][action_key]

    def save(self, path: str):
        """Save agent to file."""
        data = {
            'q_values': dict(self.q_values),
            'returns': dict(self.returns),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'grid_size': self.grid_size,
            'num_tile_types': self.num_tile_types,
            'gamma': self.gamma
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load agent from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.q_values = defaultdict(lambda: defaultdict(float), data['q_values'])
        self.returns = defaultdict(lambda: defaultdict(list), data['returns'])
        self.epsilon = data['epsilon']
        self.episodes_trained = data['episodes_trained']
        self.grid_size = data['grid_size']
        self.num_tile_types = data['num_tile_types']
        self.gamma = data['gamma']

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        num_states = len(self.q_values)
        num_state_action_pairs = sum(len(actions) for actions in self.q_values.values())

        return {
            'episodes_trained': self.episodes_trained,
            'epsilon': self.epsilon,
            'num_states': num_states,
            'num_state_action_pairs': num_state_action_pairs
        }
