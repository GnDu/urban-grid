"""
Real-time GUI visualization of DQN agent acting on a 10x10 Urban-Grid environment.
Uses pygame for interactive visualization.
"""

import pygame
import numpy as np
import torch
import sys
from gym_wrapper import UrbanGridEnv
from agents.dqn_agent import DQNAgent
from utils import TileTypes

# Colors for different tile types (RGB)
TILE_COLORS = {
    TileTypes.BARREN.value: (139, 69, 19),      # Brown
    TileTypes.RESIDENCE.value: (255, 215, 0),   # Gold
    TileTypes.GREENERY.value: (34, 139, 34),    # Green
    TileTypes.INDUSTRY.value: (128, 128, 128),  # Gray
    TileTypes.SERVICE.value: (65, 105, 225),    # Blue
    TileTypes.ROAD.value: (0, 0, 0),            # Black
}

class DQNVisualizer:
    """Real-time visualization of DQN agent."""

    def __init__(self, checkpoint_path: str, grid_size: int = 10):
        """
        Initialize the visualizer.

        Args:
            checkpoint_path: Path to the trained DQN checkpoint
            grid_size: Size of the grid (default: 10x10)
        """
        self.grid_size = grid_size
        self.cell_size = 50  # pixels per cell
        self.info_panel_width = 300

        # Window dimensions
        self.grid_width = grid_size * self.cell_size
        self.grid_height = grid_size * self.cell_size
        self.window_width = self.grid_width + self.info_panel_width
        self.window_height = self.grid_height

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("DQN Urban Grid - Real-time Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Load DQN agent
        print(f"Loading DQN agent from: {checkpoint_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = DQNAgent(
            grid_size=grid_size,
            num_tile_types=5,
            device=device
        )
        self.agent.load(checkpoint_path)
        print("Agent loaded successfully!")

        # Create environment
        self.env = UrbanGridEnv(grid_size=grid_size, pollution_coefficient=1.0)

        # Episode metrics
        self.reset_episode()

        # Control
        self.paused = False
        self.step_delay = 500  # milliseconds between steps
        self.last_step_time = 0

    def reset_episode(self):
        """Reset the episode."""
        self.state, _ = self.env.reset()
        self.episode_reward = 0
        self.step_count = 0
        self.done = False

    def draw_grid(self):
        """Draw the grid with tiles."""
        tile_grid = self.env.render()

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                tile_type = tile_grid[row, col]
                color = TILE_COLORS.get(tile_type, (255, 255, 255))

                # Draw tile
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Grid lines

    def draw_info_panel(self):
        """Draw the information panel."""
        panel_x = self.grid_width
        panel_rect = pygame.Rect(panel_x, 0, self.info_panel_width, self.window_height)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)

        # Get current info
        info = self.env._get_info()

        # Title
        title = self.font.render("DQN Agent", True, (0, 0, 0))
        self.screen.blit(title, (panel_x + 10, 10))

        # Metrics
        y_offset = 50
        metrics = [
            f"Step: {self.step_count}",
            f"Reward: {self.episode_reward:.2f}",
            "",
            f"Population: {info['total_population']:.1f}",
            f"Pollution: {info['total_pollution']:.1f}",
            f"Pop Cap: {info['population_cap']:.1f}",
            "",
            f"Pop Gain: {info['curr_pop_g']:.2f}",
            f"Poll Gain: {info['curr_poll_g']:.2f}",
            "",
            f"Barren Cells: {info['num_barren_cells']}",
            f"Epsilon: {self.agent.epsilon:.4f}",
        ]

        for metric in metrics:
            if metric:  # Skip empty strings
                text = self.small_font.render(metric, True, (0, 0, 0))
                self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 25

        # Legend
        y_offset += 20
        legend_title = self.font.render("Legend:", True, (0, 0, 0))
        self.screen.blit(legend_title, (panel_x + 10, y_offset))
        y_offset += 30

        legend_items = [
            ("Barren", TileTypes.BARREN.value),
            ("Residence", TileTypes.RESIDENCE.value),
            ("Greenery", TileTypes.GREENERY.value),
            ("Industry", TileTypes.INDUSTRY.value),
            ("Service", TileTypes.SERVICE.value),
            ("Road", TileTypes.ROAD.value),
        ]

        for name, tile_type in legend_items:
            # Color box
            color = TILE_COLORS[tile_type]
            rect = pygame.Rect(panel_x + 10, y_offset, 20, 20)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

            # Label
            text = self.small_font.render(name, True, (0, 0, 0))
            self.screen.blit(text, (panel_x + 40, y_offset))
            y_offset += 25

        # Controls
        y_offset += 20
        controls_title = self.font.render("Controls:", True, (0, 0, 0))
        self.screen.blit(controls_title, (panel_x + 10, y_offset))
        y_offset += 30

        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Episode",
            "Q: Quit",
            "UP: Speed Up",
            "DOWN: Slow Down",
        ]

        for control in controls:
            text = self.small_font.render(control, True, (0, 0, 0))
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 20

        # Status
        y_offset += 10
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 0, 0) if self.paused else (0, 200, 0)
        status_text = self.font.render(status, True, status_color)
        self.screen.blit(status_text, (panel_x + 10, y_offset))

    def step_agent(self):
        """Execute one step of the agent."""
        if self.done:
            return

        # Get valid actions
        valid_actions = self.env.get_valid_actions()

        # Select action (no exploration)
        action = self.agent.select_action(self.state, valid_actions, training=False)

        # Take step
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated

        # Update metrics
        self.episode_reward += reward
        self.step_count += 1
        self.state = next_state

        if self.done:
            print(f"\nEpisode finished!")
            print(f"  Steps: {self.step_count}")
            print(f"  Total Reward: {self.episode_reward:.2f}")
            print(f"  Final Population: {info['total_population']:.1f}")
            print(f"  Final Pollution: {info['total_pollution']:.1f}")
            print(f"  Population/Pollution Ratio: {info['total_population']/max(info['total_pollution'], 1):.2f}")

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    print("\nResetting episode...")
                    self.reset_episode()
                elif event.key == pygame.K_UP:
                    self.step_delay = max(50, self.step_delay - 100)
                    print(f"Speed: {self.step_delay}ms delay")
                elif event.key == pygame.K_DOWN:
                    self.step_delay = min(2000, self.step_delay + 100)
                    print(f"Speed: {self.step_delay}ms delay")

        return True

    def run(self):
        """Main visualization loop."""
        print("\nStarting visualization...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset Episode")
        print("  Q: Quit")
        print("  UP/DOWN: Adjust speed")
        print()

        running = True
        while running:
            current_time = pygame.time.get_ticks()

            # Handle events
            running = self.handle_events()

            # Step agent if not paused
            if not self.paused and not self.done:
                if current_time - self.last_step_time >= self.step_delay:
                    self.step_agent()
                    self.last_step_time = current_time

            # Draw
            self.screen.fill((255, 255, 255))
            self.draw_grid()
            self.draw_info_panel()
            pygame.display.flip()

            # Control frame rate
            self.clock.tick(60)

        pygame.quit()


def main():
    """Main function."""
    import os

    # Check for checkpoint
    checkpoint_path = 'checkpoints/dqn_final.pt'

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the agent first using: python train_dqn.py")
        print("\nAlternatively, specify a different checkpoint path:")
        print("  python visualize_dqn_gui.py <checkpoint_path>")
        sys.exit(1)

    # Allow custom checkpoint path from command line
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    # Create visualizer
    visualizer = DQNVisualizer(checkpoint_path, grid_size=10)

    # Run visualization
    visualizer.run()


if __name__ == '__main__':
    main()
