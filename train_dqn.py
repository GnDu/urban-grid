"""
Training script for DQN agent on Urban-Grid environment.
Uses temporal difference learning with experience replay.
"""

import numpy as np
import torch
from gym_wrapper import UrbanGridEnv
from agents.dqn_agent import DQNAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict
import os


class DQNTrainer:
    """Trainer for DQN agent on Urban-Grid environment."""

    def __init__(
        self,
        env: UrbanGridEnv,
        agent: DQNAgent,
        num_episodes: int = 1000,
        save_freq: int = 100,
        eval_freq: int = 50,
        eval_episodes: int = 5,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.eval_rewards: List[float] = []
        self.eval_populations: List[float] = []
        self.eval_pollutions: List[float] = []

    def train(self):
        """Main training loop."""
        print(f"Starting DQN training for {self.num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        print(f"Grid size: {self.env.grid_size}x{self.env.grid_size}")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space: {self.env.observation_space}")
        print("-" * 80)

        # Create progress bar
        pbar = tqdm(range(self.num_episodes), desc="Training")

        for episode in pbar:
            episode_reward, episode_length = self._train_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Update progress bar with metrics
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                pbar.set_postfix({
                    'Avg_R': f'{avg_reward:.2f}',
                    'Eps': f'{self.agent.epsilon:.4f}',
                    'Buffer': len(self.agent.replay_buffer)
                })

            # Evaluate agent
            if (episode + 1) % self.eval_freq == 0:
                eval_metrics = self._evaluate()
                self.eval_rewards.append(eval_metrics['reward'])
                self.eval_populations.append(eval_metrics['population'])
                self.eval_pollutions.append(eval_metrics['pollution'])

                tqdm.write(f"\nEvaluation @ Ep {episode + 1}: Reward={eval_metrics['reward']:.2f}, "
                          f"Pop={eval_metrics['population']:.2f}, Poll={eval_metrics['pollution']:.2f}, "
                          f"Ratio={eval_metrics['population']/max(eval_metrics['pollution'], 1):.2f}")

            # Save checkpoint
            if (episode + 1) % self.save_freq == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'dqn_episode_{episode+1}.pt')
                self.agent.save(checkpoint_path)
                tqdm.write(f"Checkpoint saved: {checkpoint_path}")

        # Final save
        final_path = os.path.join(self.checkpoint_dir, 'dqn_final.pt')
        self.agent.save(final_path)
        print(f"\nTraining complete! Final model saved: {final_path}")

        # Plot training curves
        self._plot_training_curves()

    def _train_episode(self) -> tuple[float, int]:
        """Run one training episode."""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get valid actions
            valid_actions = self.env.get_valid_actions()

            # Select action
            action = self.agent.select_action(state, valid_actions, training=True)

            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store experience
            self.agent.store_experience(state, action, reward, next_state, done)

            # Train agent
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)

            episode_reward += reward
            episode_length += 1
            state = next_state

        return episode_reward, episode_length

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate agent without exploration."""
        rewards = []
        populations = []
        pollutions = []

        for _ in range(self.eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                valid_actions = self.env.get_valid_actions()
                action = self.agent.select_action(state, valid_actions, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

            rewards.append(episode_reward)
            populations.append(info['total_population'])
            pollutions.append(info['total_pollution'])

        return {
            'reward': np.mean(rewards),
            'population': np.mean(populations),
            'pollution': np.mean(pollutions)
        }

    def _plot_training_curves(self):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) >= 10:
            smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(smoothed, label='Smoothed (window=10)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)

        # Evaluation metrics
        if self.eval_rewards:
            eval_episodes = np.arange(0, len(self.eval_rewards)) * self.eval_freq
            axes[1, 0].plot(eval_episodes, self.eval_rewards, marker='o', label='Reward')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Evaluation Reward')
            axes[1, 0].set_title('Evaluation Performance')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # Population vs Pollution
            axes[1, 1].plot(eval_episodes, self.eval_populations, marker='o', label='Population')
            axes[1, 1].plot(eval_episodes, self.eval_pollutions, marker='s', label='Pollution')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Population vs Pollution (Evaluation)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved: {plot_path}")
        plt.close()


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = UrbanGridEnv(
        grid_size=16,
        pollution_coefficient=1.0,
        seed=seed
    )

    # Create DQN agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(
        grid_size=16,
        num_tile_types=5,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=100,
        hidden_dim=256,
        device=device,
        double_dqn=True
    )

    # Create trainer
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        num_episodes=1000,
        save_freq=100,
        eval_freq=50,
        eval_episodes=5,
        checkpoint_dir='checkpoints'
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
