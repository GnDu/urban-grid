"""
Example usage of the DQN agent for Urban-Grid city planning.
Demonstrates how to load a trained agent and evaluate it against baselines.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from gym_wrapper import UrbanGridEnv
from agents.dqn_agent import DQNAgent
from agents.agent import CityPlanner
from environments.environment import CityModel
from update_rules.update_rules import DefaultUpdateRules, DefaultUpdateRulesParameters
from utils import TileTypes
import torch


class RandomCityPlanner(CityPlanner):
    """Random baseline agent for comparison."""

    def __init__(self, model, destroy_tile_penalty: float = 10):
        super().__init__(model, destroy_tile_penalty)
        self.RAND_TILES = [
            TileTypes.RESIDENCE.value,
            TileTypes.GREENERY.value,
            TileTypes.INDUSTRY.value,
            TileTypes.SERVICE.value,
            TileTypes.ROAD.value
        ]

    def decide(self):
        # Find barren cells
        barren_cells = np.where(self.model.grid.tile._mesa_data == TileTypes.BARREN.value)
        if len(barren_cells[0]) == 0:
            return

        # Random position
        idx = self.model.rng.integers(0, len(barren_cells[0]))
        x_row, y_col = barren_cells[0][idx], barren_cells[1][idx]

        # Random tile
        rand_tile = self.model.rng.choice(self.RAND_TILES)

        self.place(x_row, y_col, rand_tile)

    def update(self, **kwargs):
        self.total_population += self.model.update_rules.curr_pop_g
        self.total_pollution += self.model.update_rules.curr_poll_g


def evaluate_dqn_agent(agent: DQNAgent, num_episodes: int = 10, grid_size: int = 16):
    """Evaluate a trained DQN agent."""
    env = UrbanGridEnv(grid_size=grid_size, pollution_coefficient=1.0)

    results = {
        'rewards': [],
        'populations': [],
        'pollutions': [],
        'population_caps': [],
        'grids': []
    }

    print(f"Evaluating DQN agent over {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

        results['rewards'].append(episode_reward)
        results['populations'].append(info['total_population'])
        results['pollutions'].append(info['total_pollution'])
        results['population_caps'].append(info['population_cap'])
        results['grids'].append(env.render())

        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Pop={info['total_population']:.1f}, Poll={info['total_pollution']:.1f}")

    return results


def evaluate_random_agent(num_episodes: int = 10, grid_size: int = 16):
    """Evaluate random baseline agent."""
    results = {
        'rewards': [],
        'populations': [],
        'pollutions': [],
        'population_caps': [],
        'grids': []
    }

    print(f"Evaluating random agent over {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Create environment
        update_rules = DefaultUpdateRules()

        with open("data/update_parameters/DefaultUpdateRule.json") as f:
                default_rule_parameters = json.load(f)

        params = DefaultUpdateRulesParameters(**default_rule_parameters)

        update_rules.set_parameters(params)

        model = CityModel(
            agent_class=RandomCityPlanner,
            width=grid_size,
            height=grid_size,
            update_rules=update_rules,
            collect_rate=1.0,
            seed=episode
        )

        agent = model.get_city_planner()

        # Run episode
        for _ in range(grid_size * grid_size):
            model.step()

        # Collect results
        reward = agent.total_population - agent.total_pollution
        results['rewards'].append(reward)
        results['populations'].append(agent.total_population)
        results['pollutions'].append(agent.total_pollution)
        results['population_caps'].append(agent.population_cap)
        results['grids'].append(model.grid.tile._mesa_data.copy())

        print(f"  Episode {episode+1}: Reward={reward:.2f}, "
              f"Pop={agent.total_population:.1f}, Poll={agent.total_pollution:.1f}")

    return results


def compare_agents(dqn_results: dict, random_results: dict):
    """Compare DQN agent against random baseline."""
    print("\n" + "="*80)
    print("COMPARISON: DQN Agent vs Random Baseline")
    print("="*80)

    metrics = ['rewards', 'populations', 'pollutions', 'population_caps']

    for metric in metrics:
        dqn_mean = np.mean(dqn_results[metric])
        dqn_std = np.std(dqn_results[metric])
        random_mean = np.mean(random_results[metric])
        random_std = np.std(random_results[metric])

        improvement = ((dqn_mean - random_mean) / max(abs(random_mean), 1e-6)) * 100

        print(f"\n{metric.upper()}:")
        print(f"  DQN:    {dqn_mean:8.2f} ± {dqn_std:.2f}")
        print(f"  Random: {random_mean:8.2f} ± {random_std:.2f}")
        print(f"  Improvement: {improvement:+.1f}%")

    print("\n" + "="*80)


def visualize_comparison(dqn_results: dict, random_results: dict, save_path: str = 'comparison.png'):
    """Create visualization comparing DQN and random agents."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        ('rewards', 'Total Reward'),
        ('populations', 'Total Population'),
        ('pollutions', 'Total Pollution')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[0, idx]

        # Bar plot with error bars
        means = [np.mean(random_results[metric]), np.mean(dqn_results[metric])]
        stds = [np.std(random_results[metric]), np.std(dqn_results[metric])]
        labels = ['Random', 'DQN']

        bars = ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color=['orange', 'blue'])
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.1f}', ha='center', va='bottom')

    # Box plots
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[1, idx]
        data = [random_results[metric], dqn_results[metric]]
        bp = ax.boxplot(data, labels=['Random', 'DQN'], patch_artist=True)

        # Color boxes
        bp['boxes'][0].set_facecolor('orange')
        bp['boxes'][1].set_facecolor('blue')

        ax.set_ylabel(label)
        ax.set_title(f'{label} Distribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nComparison plot saved: {save_path}")
    plt.close()


def visualize_city_grids(dqn_results: dict, random_results: dict, num_samples: int = 3):
    """Visualize example city grids from both agents."""
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))

    # Colormap for tiles
    cmap = plt.cm.get_cmap('tab10')

    for i in range(num_samples):
        # DQN grid
        ax_dqn = axes[0, i]
        dqn_grid = dqn_results['grids'][i]
        im = ax_dqn.imshow(dqn_grid, cmap=cmap, vmin=0, vmax=5)
        ax_dqn.set_title(f'DQN Episode {i+1}\n'
                         f'Pop: {dqn_results["populations"][i]:.1f}, '
                         f'Poll: {dqn_results["pollutions"][i]:.1f}')
        ax_dqn.axis('off')

        # Random grid
        ax_random = axes[1, i]
        random_grid = random_results['grids'][i]
        im = ax_random.imshow(random_grid, cmap=cmap, vmin=0, vmax=5)
        ax_random.set_title(f'Random Episode {i+1}\n'
                           f'Pop: {random_results["populations"][i]:.1f}, '
                           f'Poll: {random_results["pollutions"][i]:.1f}')
        ax_random.axis('off')

    # Add colorbar legend
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                       fraction=0.046, pad=0.04)
    cbar.set_label('Tile Types: 0=Barren, 1=Residence, 2=Greenery, 3=Industry, 4=Service, 5=Road')

    plt.tight_layout()
    plt.savefig('city_grids_comparison.png', dpi=150)
    print("City grids comparison saved: city_grids_comparison.png")
    plt.close()


def main():
    """Main evaluation function."""
    print("="*80)
    print("DQN Agent Evaluation for Urban-Grid City Planning")
    print("="*80)

    grid_size = 16
    num_episodes = 10

    # Option 1: Load a trained agent
    checkpoint_path = 'checkpoints/dqn_final.pt'

    try:
        print(f"\nLoading trained DQN agent from: {checkpoint_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = DQNAgent(
            grid_size=grid_size,
            num_tile_types=5,
            device=device
        )
        agent.load(checkpoint_path)
        print("Agent loaded successfully!")

        # Evaluate DQN agent
        print("\n" + "-"*80)
        dqn_results = evaluate_dqn_agent(agent, num_episodes=num_episodes, grid_size=grid_size)

    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the agent first using: python train_dqn.py")
        print("\nFor demonstration, evaluating random agent only...")
        dqn_results = None

    # Evaluate random baseline
    print("\n" + "-"*80)
    random_results = evaluate_random_agent(num_episodes=num_episodes, grid_size=grid_size)

    # Compare results
    if dqn_results is not None:
        compare_agents(dqn_results, random_results)
        visualize_comparison(dqn_results, random_results)
        visualize_city_grids(dqn_results, random_results, num_samples=min(3, num_episodes))

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
