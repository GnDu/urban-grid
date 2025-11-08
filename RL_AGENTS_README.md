# RL Agents for Urban-Grid City Planning

This directory contains reinforcement learning agents for the Urban-Grid city planning simulation. Two RL approaches are implemented: **Deep Q-Network (DQN)** using temporal difference learning, and **Monte Carlo** methods using episodic learning.

## Overview

The Urban-Grid environment is a grid-based city simulation where an agent acts as a city planner, placing tiles (residences, industries, services, greenery, roads) to maximize population growth while minimizing pollution. The environment provides:

- **State Space**: Grid layouts (tile types, population gains, pollution gains) + scalar features
- **Action Space**: Place a tile at position (row, col) with type (RESIDENCE, GREENERY, INDUSTRY, SERVICE, ROAD)
- **Reward**: Population gain - Pollution gain
- **Episode Length**: Until grid is full (default: 16×16 = 256 steps)

## Files Structure

```
urban-grid/
├── gym_wrapper.py              # Gymnasium environment wrapper
├── agents/
│   ├── dqn_agent.py           # Deep Q-Network agent (Temporal Difference)
│   └── monte_carlo_agent.py   # Monte Carlo agent (Episodic learning)
├── train_dqn.py               # Training script for DQN
├── train_monte_carlo.py       # Training script for Monte Carlo
├── example_dqn_agent.py       # Evaluation and comparison script
└── RL_AGENTS_README.md        # This file
```

## Installation

### Requirements

```bash
pip install gymnasium numpy torch matplotlib tqdm scipy mesa
```

Or install from the project's requirements file:

```bash
pip install -r requirements.txt
```

### Dependencies

- **gymnasium**: OpenAI Gym interface
- **torch**: PyTorch for DQN neural networks
- **numpy**: Numerical operations
- **matplotlib**: Visualization
- **mesa**: Agent-based modeling framework (existing dependency)
- **scipy**: Scientific computing (existing dependency)

## Agent 1: Deep Q-Network (DQN)

### Algorithm: Temporal Difference Learning

DQN uses **temporal difference (TD) learning**, specifically the Bellman equation:

```
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

This is a **one-step TD method** that updates Q-values after each action, making it more sample-efficient than Monte Carlo.

### Features

- **Experience Replay**: Stores transitions in a replay buffer and samples mini-batches for training
- **Target Network**: Separate network for stable Q-value targets, updated periodically
- **Double DQN**: Uses policy network to select actions and target network to evaluate them (reduces overestimation)
- **Convolutional Architecture**: Processes grid-based observations with CNN layers
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation

### Architecture

```
Input:
  - Grid layers (3 channels): tile_grid, pop_g_grid, poll_g_grid
  - Scalar features (4 values): time_step, total_population, total_pollution, population_cap

Network:
  - Conv2D(3→32→64→64) for grid processing
  - MLP(4→64→128) for scalar features
  - Combined MLP(combined→256→256)
  - Output: Q-values for all (row, col, tile_type) actions

Output: Q(s, a) for each possible action
```

### Training

```bash
python train_dqn.py
```

**Default hyperparameters:**
- Learning rate: 1e-4
- Gamma (discount): 0.99
- Epsilon: 1.0 → 0.05 (decay: 0.995)
- Batch size: 64
- Replay buffer: 100,000
- Target update frequency: 100 steps
- Episodes: 1000

**Training time**: ~2-4 hours on CPU, ~30-60 minutes on GPU (for 1000 episodes)

### Checkpoints

Models are saved to `checkpoints/`:
- `dqn_episode_100.pt`, `dqn_episode_200.pt`, ...: Periodic checkpoints
- `dqn_final.pt`: Final trained model

## Agent 2: Monte Carlo (MC)

### Algorithm: Episodic Learning

Monte Carlo uses **complete episode returns** to update Q-values:

```
Q(s, a) ← average of returns G_t following (s, a)
```

where `G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^T r_T` (sum of discounted rewards).

This is an **every-episode update** method that waits until the episode completes before learning.

### Features

- **First-Visit or Every-Visit MC**: Can use first-visit or every-visit variant
- **Simplified State Representation**: Uses discretized features to make tabular Q-learning tractable
- **Epsilon-Greedy Exploration**: Same exploration strategy as DQN
- **No Neural Network**: Direct Q-table stored in dictionary (memory-efficient for small state spaces)

### State Representation (Simplified)

To make tabular Q-learning tractable, the MC agent uses a simplified state:

```
State = (tile_counts[0-5], time_step, total_pop_bin, total_poll_bin, pop_cap_bin)
```

Where continuous values are discretized into bins (e.g., population // 10).

### Training

```bash
python train_monte_carlo.py
```

**Default hyperparameters:**
- Gamma (discount): 0.99
- Epsilon: 1.0 → 0.05 (decay: 0.995)
- First-visit: True
- Episodes: 1000

**Training time**: ~1-2 hours on CPU (for 1000 episodes)

### Checkpoints

Models are saved to `checkpoints_mc/`:
- `mc_episode_100.pkl`, `mc_episode_200.pkl`, ...: Periodic checkpoints
- `mc_final.pkl`: Final trained model

## Evaluation and Comparison

### Running Evaluation

```bash
python example_dqn_agent.py
```

This script:
1. Loads a trained DQN agent (if available)
2. Evaluates DQN agent over multiple episodes
3. Evaluates random baseline agent
4. Compares performance metrics
5. Generates visualizations

### Metrics

- **Total Reward**: Population gain - Pollution gain
- **Total Population**: Final population achieved
- **Total Pollution**: Final pollution level
- **Population Cap**: Maximum population capacity (based on residences)

### Visualizations

Generated plots:
- `comparison.png`: Bar charts and box plots comparing DQN vs Random
- `city_grids_comparison.png`: Side-by-side city layouts from both agents
- `training_curves.png`: Training progress over episodes

## Usage Example

### Training DQN Agent

```python
from gym_wrapper import UrbanGridEnv
from agents.dqn_agent import DQNAgent
import torch

# Create environment
env = UrbanGridEnv(grid_size=16, pollution_coefficient=1.0)

# Create agent
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = DQNAgent(
    grid_size=16,
    num_tile_types=5,
    learning_rate=1e-4,
    device=device
)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_experience(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state

# Save agent
agent.save('my_agent.pt')
```

### Using Trained Agent

```python
from gym_wrapper import UrbanGridEnv
from agents.dqn_agent import DQNAgent

# Load agent
agent = DQNAgent(grid_size=16, num_tile_types=5)
agent.load('checkpoints/dqn_final.pt')

# Evaluate
env = UrbanGridEnv(grid_size=16)
state, _ = env.reset()
done = False

while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions, training=False)  # Greedy
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Final Population: {info['total_population']}")
print(f"Final Pollution: {info['total_pollution']}")
```

## Comparison: DQN vs Monte Carlo

| Feature | DQN (Temporal Difference) | Monte Carlo (Episodic) |
|---------|---------------------------|------------------------|
| **Learning Method** | One-step TD updates | Full episode returns |
| **Sample Efficiency** | More efficient (online learning) | Less efficient (needs full episodes) |
| **Memory** | Neural network (~1-5 MB) | Q-table (grows with states) |
| **State Representation** | Full grid observation | Simplified discretized state |
| **Training Speed** | Slower per episode (NN updates) | Faster per episode (table lookup) |
| **Scalability** | Scales to large state spaces | Limited by state space size |
| **Convergence** | Asymptotically converges | Guaranteed convergence (tabular) |
| **Best For** | Complex, large-scale problems | Smaller, episodic problems |

### When to Use Each

**Use DQN when:**
- State space is large or continuous
- You need sample efficiency
- You have GPU resources
- You want deep feature learning

**Use Monte Carlo when:**
- State space is small/discrete
- Episodes are short
- You want simpler, interpretable learning
- You have limited computational resources

## Customization

### Modify Reward Function

Edit `gym_wrapper.py`, line ~120:

```python
# Current: reward = pop_delta - pollution_coefficient * poll_delta
# Custom: reward = pop_delta - 0.5 * poll_delta + 0.1 * pop_cap_delta
```

### Change Grid Size

Modify `train_dqn.py` or `train_monte_carlo.py`:

```python
env = UrbanGridEnv(grid_size=32)  # Larger grid
agent = DQNAgent(grid_size=32, ...)
```

### Adjust Hyperparameters

```python
agent = DQNAgent(
    learning_rate=5e-4,      # Faster learning
    gamma=0.95,              # Less long-term focus
    epsilon_decay=0.99,      # Slower exploration decay
    batch_size=128,          # Larger batches
    hidden_dim=512           # Bigger network
)
```

## Troubleshooting

### DQN Training Issues

1. **Loss exploding**: Reduce learning rate, add gradient clipping
2. **No improvement**: Increase exploration (higher epsilon_end), check reward scaling
3. **Slow training**: Use GPU, reduce batch size, decrease target_update_freq
4. **Out of memory**: Reduce buffer_size, batch_size, or hidden_dim

### Monte Carlo Training Issues

1. **State space too large**: Simplify state representation further
2. **Slow convergence**: Increase episodes, adjust epsilon decay
3. **Memory issues**: Use first-visit MC, limit state discretization

## Expected Performance

After 1000 training episodes:

| Metric | Random Baseline | DQN (Expected) | MC (Expected) |
|--------|----------------|----------------|---------------|
| **Avg Reward** | 50-100 | 150-250 | 100-180 |
| **Avg Population** | 200-300 | 350-450 | 280-380 |
| **Avg Pollution** | 150-250 | 150-220 | 150-230 |

Note: Actual performance varies with hyperparameters and training time.

## References

### DQN (Temporal Difference Learning)
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
- Van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.

### Monte Carlo Methods
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction." Chapter 5.

### Urban Planning & RL
- Zheng et al. (2018). "Learning to Optimize City Planning." *ACM SIGSPATIAL*.

## License

This code is part of the Urban-Grid project. See main repository for license details.

## Contact

For questions or issues, please open an issue in the GitHub repository.
