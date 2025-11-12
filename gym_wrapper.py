"""
Gymnasium wrapper for the Urban-Grid city simulation environment.
Provides a standard RL interface for training agents.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from environments.environment import CityModel
from update_rules.update_rules import DefaultUpdateRules, DefaultUpdateRulesParameters
from utils import TileTypes
from agents.agent import CityPlanner
import json

class DummyAgent(CityPlanner):
    """Dummy agent that does nothing - controlled externally by Gym wrapper."""

    def decide(self):
        # No-op: actions handled by gym wrapper
        pass

    def update(self, **kwargs):
        self.total_population += self.model.update_rules.curr_pop_g
        self.total_pollution += self.model.update_rules.curr_poll_g
        self.population_cap = self.model.update_rules.population_cap


class UrbanGridEnv(gym.Env):
    """
    Gymnasium environment for Urban-Grid city planning simulation.

    Observation Space:
        - Grid layers: tile types, population gains, pollution gains
        - Scalar features: time_step, total_population, total_pollution, population_cap

    Action Space:
        - MultiDiscrete: [row, col, tile_type]
        - row/col: position on grid (0 to grid_size-1)
        - tile_type: 1-5 (RESIDENCE, GREENERY, INDUSTRY, SERVICE, ROAD)

    Reward:
        - population_gain - pollution_coefficient * pollution_gain
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 16,
        pollution_coefficient: float = 1.0,
        max_steps: Optional[int] = None,
        update_rules: Optional[DefaultUpdateRules] = None
    ):
        super().__init__()

        self.grid_size = grid_size
        self.pollution_coefficient = pollution_coefficient
        self.max_steps = max_steps or (grid_size * grid_size)  # One tile per cell

        # Setup update rules
        if update_rules is None:
            self.update_rules = DefaultUpdateRules()
                    
            with open("data/update_parameters/DefaultUpdateRule.json") as f:
                default_rule_parameters = json.load(f)

            params = DefaultUpdateRulesParameters(**default_rule_parameters)
            self.update_rules.set_parameters(params)
        else:
            self.update_rules = update_rules

        # Define action space: [row, col, tile_type]
        # tile_type: 1=RESIDENCE, 2=GREENERY, 3=INDUSTRY, 4=SERVICE, 5=ROAD
        self.action_space = gym.spaces.MultiDiscrete([grid_size, grid_size, 5])

        # Define observation space
        # 5 grid layers (tiles, pop_g, poll_g, block_id, road_connectivity) + 4 scalar features
        self.observation_space = gym.spaces.Dict({
            "tile_grid": gym.spaces.Box(
                low=0, high=5, shape=(grid_size, grid_size), dtype=np.int32
            ),
            "pop_g_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(grid_size, grid_size), dtype=np.float32
            ),
            "poll_g_grid": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(grid_size, grid_size), dtype=np.float32
            ),
            "block_id_grid": gym.spaces.Box(
                low=0, high=np.inf, shape=(grid_size, grid_size), dtype=np.int32
            ),
            "road_connectivity_grid": gym.spaces.Box(
                low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
            ),
            "features": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )
        })

        self.model: Optional[CityModel] = None
        self.agent: Optional[DummyAgent] = None
        self.prev_population = 0
        self.prev_pollution = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: If provided, uses this specific seed (for evaluation/reproducibility).
                  If None, generates a new random seed each episode (for training variety).
        """
        super().reset(seed=seed)

        # Use provided seed for evaluation, or generate new random seed for training
        if seed is not None:
            episode_seed = seed
        else:
            # Generate a new random seed for this episode (ensures variety during training)
            episode_seed = np.random.randint(0, 2**31 - 1)

        # Create new city model with the episode seed
        self.model = CityModel(
            agent_class=DummyAgent,
            width=self.grid_size,
            height=self.grid_size,
            update_rules=self.update_rules,
            collect_rate=1.0,
            seed=episode_seed
        )

        self.agent = self.model.get_city_planner()
        self.prev_population = 0
        self.prev_pollution = 0
        self._prev_connected_blocks = 0  # Track connected blocks for reward shaping

        # Initialize random edge with roads
        self._initialize_edge_roads()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _initialize_edge_roads(self):
        """
        Place a single road tile randomly at the border of the map.
        Uses Mesa's random generator (self.model.random) for reproducibility with seeds.
        """
        # Collect all border positions
        border_positions = []

        # Top edge (row 0)
        for col in range(self.grid_size):
            border_positions.append((0, col))

        # Bottom edge (row = grid_size - 1)
        for col in range(self.grid_size):
            border_positions.append((self.grid_size - 1, col))

        # Left edge (col 0), excluding corners already added
        for row in range(1, self.grid_size - 1):
            border_positions.append((row, 0))

        # Right edge (col = grid_size - 1), excluding corners already added
        for row in range(1, self.grid_size - 1):
            border_positions.append((row, self.grid_size - 1))

        # Randomly select one border position using Mesa's random generator
        idx = int(self.model.random.random() * len(border_positions))
        row, col = border_positions[idx]

        # Place single road tile
        self.agent.place(row, col, TileTypes.ROAD.value)

        self.model.book_keep()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [row, col, tile_type] where tile_type is 0-4 (maps to TILE_TYPES 1-5)

        Returns:
            observation, reward, terminated, truncated, info
        """
        row, col, tile_type_idx = action
        tile_type = tile_type_idx + 1  # Convert 0-4 to 1-5 (skip BARREN=0)

        # Store previous state for reward calculation
        prev_pop = self.agent.total_population
        prev_poll = self.agent.total_pollution

        # Execute action - allow rewriting existing tiles
        try:
            cell_value = self.model.grid.tile._mesa_data[row, col]

            # Small penalty for rewriting existing tiles (to encourage efficient placement)
            rewrite_penalty = 0.0
            if cell_value != TileTypes.BARREN.value:
                rewrite_penalty = -1.0  # Small cost for changing existing tiles

            # Place the tile (will overwrite if not barren)
            self.agent.place(row, col, tile_type)

        except Exception as e:
            # Action execution failed - penalize
            reward = -10.0
            observation = self._get_observation()
            info = self._get_info()
            info['action_error'] = str(e)
            return observation, reward, False, False, info

        # Apply game rules
        self.model.book_keep()
        self.model.update_rules.apply_rules(self.model)
        self.agent.update()

        # Collect data
        if self.model.time_step % self.model.collect_rate == 0:
            self.model.data_collectors.collect(self.model)

        self.model.time_step += 1

        # Calculate reward with improved shaping for balanced city building
        pop_delta = self.agent.total_population - prev_pop
        poll_delta = self.agent.total_pollution - prev_poll

        # Base reward: population gain - pollution gain
        base_reward = pop_delta - self.pollution_coefficient * poll_delta

        # CRITICAL: Direct immediate bonus for placing roads (positive reinforcement)
        road_placement_bonus = 0.0
        if tile_type == TileTypes.ROAD.value:
            road_placement_bonus = 3.0  # Strong immediate reward for building roads

        # Strong bonus for connecting blocks to roads
        num_road_connected_blocks = len(self.model.road_connected_blocks)
        prev_connected = getattr(self, '_prev_connected_blocks', 0)
        newly_connected = num_road_connected_blocks - prev_connected
        self._prev_connected_blocks = num_road_connected_blocks

        # Massive reward for connecting new blocks (this is the key incentive!)
        connection_bonus = newly_connected * 10.0  # Increased from 5.0 to 10.0

        # Ongoing bonus for maintaining connected blocks
        block_bonus = num_road_connected_blocks * 0.5

        # Calculate average block size (reward larger, more efficient blocks)
        total_tiles_in_blocks = 0
        for tile_type_enum, blocks_dict in self.model.blocks_by_type.items():
            for block_id, block_cells in blocks_dict.items():
                if block_id in self.model.road_connected_blocks:
                    total_tiles_in_blocks += len(block_cells)

        avg_block_size = total_tiles_in_blocks / max(num_road_connected_blocks, 1)
        size_bonus = (avg_block_size - 1) * 0.1

        # Bonus for non-road tiles placed adjacent to roads (strategic placement)
        adjacency_bonus = 0.0
        if tile_type != TileTypes.ROAD.value and self._is_adjacent_to_road(row, col):
            adjacency_bonus = 1.0  # Reward for building near road infrastructure

        # Road network size bonus (encourages building connected road networks)
        num_roads = np.count_nonzero(self.model.road_tiles)
        road_network_bonus = np.sqrt(num_roads) * 0.3  # Sublinear scaling

        # Count total non-road, non-barren buildings
        total_buildings = 0
        for tile_type_val in [TileTypes.RESIDENCE.value, TileTypes.GREENERY.value,
                               TileTypes.INDUSTRY.value, TileTypes.SERVICE.value]:
            total_buildings += np.count_nonzero(self.model.grid.tile._mesa_data == tile_type_val)

        # Connectivity efficiency bonus: reward high ratio of connected buildings
        if total_buildings > 0:
            connectivity_efficiency = total_tiles_in_blocks / total_buildings
            efficiency_bonus = connectivity_efficiency * 2.0  # Bonus for high connectivity
        else:
            efficiency_bonus = 0.0

        # Gentler road penalty (only penalize excessive roads)
        num_non_barren = self.grid_size * self.grid_size - len(np.where(self.model.grid.tile._mesa_data == TileTypes.BARREN.value)[0])
        if num_non_barren > 0 and num_roads > 0:
            road_ratio = num_roads / num_non_barren
            # Only penalize if roads exceed 40% of tiles (down from 50%)
            road_penalty = -road_ratio * 2.0 if road_ratio > 0.4 else 0
        else:
            road_penalty = 0

        # Population cap utilization bonus (encourage building residences)
        current_pop = self.agent.total_population
        pop_cap = self.agent.population_cap
        if pop_cap > 0:
            utilization = current_pop / pop_cap
            # Bonus for approaching capacity (but not exceeding)
            cap_bonus = min(utilization, 1.0) * 2.0  # Up to +2.0 for full utilization
        else:
            cap_bonus = 0.0

        # Count each building type for ratio-based balancing
        num_residences = np.count_nonzero(self.model.grid.tile._mesa_data == TileTypes.RESIDENCE.value)
        num_greenery = np.count_nonzero(self.model.grid.tile._mesa_data == TileTypes.GREENERY.value)
        num_industry = np.count_nonzero(self.model.grid.tile._mesa_data == TileTypes.INDUSTRY.value)
        num_service = np.count_nonzero(self.model.grid.tile._mesa_data == TileTypes.SERVICE.value)

        # Ratio-based balance incentives
        balance_penalty = 0.0
        if num_residences > 0:
            # Ideal ratios: 1 greenery per 2-3 residences, 1 industry per 3-4 residences, 1 service per 4-5 residences
            greenery_ratio = num_greenery / num_residences
            industry_ratio = num_industry / num_residences
            service_ratio = num_service / num_residences

            # Penalize too many greenery (optimal: 0.3-0.5, i.e., 1 per 2-3 residences)
            if greenery_ratio > 0.6:
                balance_penalty += (greenery_ratio - 0.6) * -3.0  # Penalty for greenery spam

            # Penalize too few greenery (need some for pollution control)
            if greenery_ratio < 0.2 and num_residences >= 5:
                balance_penalty += (0.2 - greenery_ratio) * -2.0

            # Penalize too many industry relative to residences
            if industry_ratio > 0.4:  # Max ~1 industry per 2.5 residences
                balance_penalty += (industry_ratio - 0.4) * -2.0

            # Penalize too many services relative to residences
            if service_ratio > 0.3:  # Max ~1 service per 3.3 residences
                balance_penalty += (service_ratio - 0.3) * -2.0

            # Bonus for balanced city (has all building types in reasonable proportions)
            has_all_types = (num_greenery > 0 and num_industry > 0 and
                           num_service > 0 and num_residences >= 3)
            if has_all_types:
                # Check if ratios are within good ranges
                good_greenery = 0.2 <= greenery_ratio <= 0.5
                good_industry = 0.1 <= industry_ratio <= 0.35
                good_service = 0.1 <= service_ratio <= 0.25

                if good_greenery and good_industry and good_service:
                    balance_penalty += 3.0  # Bonus for well-balanced city

        # Combine all reward components
        reward = (base_reward + road_placement_bonus + connection_bonus +
                 block_bonus + size_bonus + adjacency_bonus +
                 road_network_bonus + efficiency_bonus + road_penalty +
                 rewrite_penalty + cap_bonus + balance_penalty)

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.model.time_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from the environment."""
        tile_grid = self.model.grid.tile._mesa_data.astype(np.int32)
        pop_g_grid = self.model.grid.pop_g._mesa_data.astype(np.float32)
        poll_g_grid = self.model.grid.poll_g._mesa_data.astype(np.float32)
        block_id_grid = self.model.block_ids.astype(np.int32)
        road_connectivity_grid = self.model.get_road_connectivity_grid()

        # Scalar features: [time_step, total_population, total_pollution, population_cap]
        features = np.array([
            self.model.time_step,
            self.agent.total_population,
            self.agent.total_pollution,
            self.agent.population_cap
        ], dtype=np.float32)

        return {
            "tile_grid": tile_grid,
            "pop_g_grid": pop_g_grid,
            "poll_g_grid": poll_g_grid,
            "block_id_grid": block_id_grid,
            "road_connectivity_grid": road_connectivity_grid,
            "features": features
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        barren_cells = np.where(self.model.grid.tile._mesa_data == TileTypes.BARREN.value)
        num_barren = len(barren_cells[0])

        return {
            "time_step": self.model.time_step,
            "total_population": self.agent.total_population,
            "total_pollution": self.agent.total_pollution,
            "population_cap": self.agent.population_cap,
            "num_barren_cells": num_barren,
            "curr_pop_g": self.model.update_rules.curr_pop_g,
            "curr_poll_g": self.model.update_rules.curr_poll_g
        }

    def _is_terminated(self) -> bool:
        """Check if episode should terminate. With rewrites enabled, rely on max_steps."""
        # Since we allow rewrites, episodes continue until max_steps
        return False

    def render(self) -> Optional[np.ndarray]:
        """Render the environment (optional)."""
        if self.model is None:
            return None

        # Return the tile grid as a simple visualization
        return self.model.grid.tile._mesa_data.copy()

    def get_valid_actions(self) -> np.ndarray:
        """
        Get list of valid actions with road adjacency constraint.
        Roads can only be placed adjacent to existing roads.
        Other tiles can be placed or rewritten on any cell.
        """
        valid_actions = []

        # Allow actions on ALL cells (enables rewriting)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Check if this cell is adjacent to a road
                is_adjacent_to_road = self._is_adjacent_to_road(row, col)

                for tile_type in range(5):  # 0-4 maps to tile types 1-5
                    actual_tile_type = tile_type + 1  # Convert to 1-5

                    # If placing a road (type 5), must be adjacent to existing road
                    if actual_tile_type == TileTypes.ROAD.value:
                        if is_adjacent_to_road:
                            valid_actions.append([row, col, tile_type])
                    else:
                        # Non-road tiles can be placed anywhere (including rewrites)
                        valid_actions.append([row, col, tile_type])

        return np.array(valid_actions) if valid_actions else np.array([]).reshape(0, 3)

    def _is_adjacent_to_road(self, row: int, col: int) -> bool:
        """Check if a cell is adjacent (4-connected) to a road."""
        # Check 4-connected neighbors
        neighbors = [
            (row - 1, col),  # up
            (row + 1, col),  # down
            (row, col - 1),  # left
            (row, col + 1)   # right
        ]

        for n_row, n_col in neighbors:
            if 0 <= n_row < self.grid_size and 0 <= n_col < self.grid_size:
                if self.model.grid.tile._mesa_data[n_row, n_col] == TileTypes.ROAD.value:
                    return True

        return False

    def close(self):
        """Clean up resources."""
        self.model = None
        self.agent = None