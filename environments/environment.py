# environments/environment.py
import mesa
import networkx as nx
import numpy as np
from collections import defaultdict
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import PropertyLayer
from scipy.ndimage import grey_dilation, generate_binary_structure, iterate_structure

from utils import TileTypes


def get_city_layout(model):
    return model.grid.tile._mesa_data

def get_population_gain(model):
    return model.update_rules.curr_pop_g

def get_pollution_gain(model):
    return model.update_rules.curr_poll_g

def get_residence_tiles_count(model):
    return np.count_nonzero(model.residence_tiles)

def get_greenery_tiles_count(model):
    return np.count_nonzero(model.greenery_tiles)

def get_industry_tiles_count(model):
    return np.count_nonzero(model.industry_tiles)

def get_service_tiles_count(model):
    return np.count_nonzero(model.service_tiles)

def get_road_tiles_count(model):
    return np.count_nonzero(model.road_tiles)

def get_population_gain_grid(model):
    return model.grid.pop_g._mesa_data

def get_pollution_gain_grid(model):
    return model.grid.poll_g._mesa_data


class CityModel(mesa.Model):
    """
    City simulator with RL hooks:
      - get_observation() -> (H, W, C) numeric
      - compute_reward(action) -> float
      - is_done() -> bool
      - valid_actions() -> iterable of ints
      - apply_action(action:int) -> mutates state and advances one tick
    """
    ACTION_TILE_TYPES = [
        TileTypes.RESIDENCE.value,
        TileTypes.GREENERY.value,
        TileTypes.INDUSTRY.value,
        TileTypes.SERVICE.value,
        TileTypes.ROAD.value,
    ]

    def __init__(self, agent_class, width, height, update_rules, collect_rate=1.0, seed=None):
        super().__init__(seed=seed)
        self.time_step = 0
        self.width = width
        self.height = height
        self.agent_class = agent_class
        self.update_rules = update_rules

        self.curr_pop_g = 0
        self.curr_poll_g = 0
        self.population_cap = 0

        # Create city planner(s)
        self.agent_class.create_agents(model=self, n=1)

        # Road network structures
        self.road_set_id = 1
        self.road_sets = np.zeros((self.width, self.height))
        self.road_adj_to_residence = defaultdict(set)
        self.road_adj_to_industries = defaultdict(set)
        self.road_adj_to_services = defaultdict(set)
        self.road_graph = nx.Graph()
        self.road_connected_sets = []

        # Grid & property layers
        self.grid = OrthogonalMooreGrid(dimensions=(self.width, self.height), random=self.random)

        tile_property_layer = PropertyLayer(
            name="tile",
            dimensions=(self.width, self.height),
            default_value=np.int8(0),
            dtype=np.int8,
        )
        pop_g_layer = PropertyLayer(
            name="pop_g",
            dimensions=(self.width, self.height),
            default_value=np.float64(0.0),
            dtype=np.float64,
        )
        poll_g_layer = PropertyLayer(
            name="poll_g",
            dimensions=(self.width, self.height),
            default_value=np.float64(0.0),
            dtype=np.float64,
        )

        self.grid.add_property_layer(tile_property_layer)
        self.grid.add_property_layer(pop_g_layer)
        self.grid.add_property_layer(poll_g_layer)

        # Book-keeping masks
        self.residence_tiles = np.zeros(self.grid.tile.dimensions)
        self.greenery_tiles = np.zeros(self.grid.tile.dimensions)
        self.industry_tiles = np.zeros(self.grid.tile.dimensions)
        self.service_tiles = np.zeros(self.grid.tile.dimensions)
        self.road_tiles = np.zeros(self.grid.tile.dimensions)

        # Data collection
        self.collect_rate = collect_rate
        self.data_collectors = mesa.DataCollector(
            model_reporters={
                "City": get_city_layout,
                "Population Gain": get_population_gain,
                "Pollution Gain": get_pollution_gain,
                "Population Gain Grid": get_population_gain_grid,
                "Pollution Gain Grid": get_pollution_gain_grid,
                "Total Residence": get_residence_tiles_count,
                "Total Industries": get_industry_tiles_count,
                "Total Greenery": get_greenery_tiles_count,
                "Total Service": get_service_tiles_count,
                "Total Road": get_road_tiles_count,
            },
            agent_reporters={
                "Total Population": "total_population",
                "Total Pollution": "total_pollution",
                "Population Cap": "population_cap",
            },
        )

    # -----------------
    # Simulation step
    # -----------------
    def step(self):
        self.agents.do("decide")
        self.book_keep()
        # Update environment based on agent decision
        self.update_rules.apply_rules(self)
        # Safe agent update (ignore abstract NotImplementedError)
        self._safe_agent_update()
        # Collect data
        if self.time_step % self.collect_rate == 0:
            self.data_collectors.collect(self)
        self.time_step += 1

    def _safe_agent_update(self):
        """Call agent.update() if present; ignore abstract NotImplementedError."""
        try:
            agent = self.agents[0]
        except Exception:
            return
        upd = getattr(agent, "update", None)
        if callable(upd):
            try:
                upd()
            except NotImplementedError:
                # Abstract base implementation — ignore
                pass

    def book_keep(self):
        # Update tile masks for quick access
        self.residence_tiles = self.grid.tile.select_cells(
            lambda data: data == TileTypes.RESIDENCE.value, return_list=False
        ).astype(int)
        self.greenery_tiles = self.grid.tile.select_cells(
            lambda data: data == TileTypes.GREENERY.value, return_list=False
        ).astype(int)
        self.industry_tiles = self.grid.tile.select_cells(
            lambda data: data == TileTypes.INDUSTRY.value, return_list=False
        ).astype(int)
        self.service_tiles = self.grid.tile.select_cells(
            lambda data: data == TileTypes.SERVICE.value, return_list=False
        ).astype(int)
        self.road_tiles = self.grid.tile.select_cells(
            lambda data: data == TileTypes.ROAD.value, return_list=False
        ).astype(int)

    # ---------------
    # Convenience API
    # ---------------
    def get_city_planner(self):
        return self.agents[0]

    def get_tile(self, row_x, col_y):
        return self.grid.tile._mesa_data[(row_x, col_y)]

    def set_tile(self, row_x, col_y, tile):
        # Directly write to property layer
        self.grid.tile._mesa_data[row_x, col_y] = tile

        if tile == TileTypes.ROAD.value:
            self.update_road_network(row_x, col_y)
        elif tile == TileTypes.RESIDENCE.value:
            self.update_residence_adjacencies(row_x, col_y)
        elif tile == TileTypes.INDUSTRY.value:
            self.update_industry_adjacencies(row_x, col_y)
        elif tile == TileTypes.SERVICE.value:
            self.update_service_adjacencies(row_x, col_y)
        elif tile == TileTypes.BARREN.value:
            # Deleting a tile
            tile_query = self.get_tile(row_x, col_y)
            tile_of_interests = [
                TileTypes.RESIDENCE.value,
                TileTypes.INDUSTRY.value,
                TileTypes.SERVICE.value,
            ]
            adj_of_interests = [
                self.road_adj_to_residence,
                self.road_adj_to_industries,
                self.road_adj_to_services,
            ]

            if tile_query in tile_of_interests:
                # Remove from adjacency lists
                interest_index = tile_of_interests.index(tile_query)
                adj_list = adj_of_interests[interest_index]
                self.remove_tile_from_adjacencies(row_x, col_y, adj_list)
            elif tile_query == TileTypes.ROAD.value:
                # Destroy road tile and relabel
                self.remove_road_tile(row_x, col_y)

    def remove_tile_from_adjacencies(self, row_x, col_y, adj_list):
        neighbours, _ = self.get_neighbours(self.road_sets, row_x, col_y, is_4_neighbourhood=True)
        neighbours = set(neighbours.flatten().tolist())
        try:
            neighbours.remove(0)
        except KeyError:
            pass
        for neighbour in neighbours:
            adj_list[neighbour].remove((row_x, col_y))

    def update_industry_adjacencies(self, row_x, col_y):
        self.update_adjacencies_to_road(row_x, col_y, TileTypes.INDUSTRY.value, self.road_adj_to_industries)

    def update_service_adjacencies(self, row_x, col_y):
        self.update_adjacencies_to_road(row_x, col_y, TileTypes.SERVICE.value, self.road_adj_to_services)

    def update_residence_adjacencies(self, row_x, col_y):
        self.update_adjacencies_to_road(row_x, col_y, TileTypes.RESIDENCE.value, self.road_adj_to_residence)

    def update_adjacencies_to_road(self, row_x, col_y, tile_value: int, road_adj_dict: defaultdict):
        """
        Given a coordinate (row_x, col_y) for a tile of type `tile_value`,
        check if there are any road networks adjacent (4-neighbourhood).
        If so, update the adjacency index for those road network IDs.
        """
        assert int(self.grid.tile._mesa_data[(row_x, col_y)]) == tile_value

        neighbours, _ = self.get_neighbours(self.road_sets, row_x, col_y, is_4_neighbourhood=True)
        neighbours = set(neighbours.flatten().tolist())
        try:
            neighbours.remove(0)
        except KeyError:
            pass

        for neighbour in neighbours:
            road_adj_dict[neighbour].add((row_x, col_y))

    def update_road_adjacencies(self, row_x, col_y, tiletype_value, curr_road_id):
        tiles = self.grid.tile.select_cells(lambda data: data == tiletype_value, return_list=False).astype(int)
        neighbours, center = self.get_neighbours(tiles, row_x, col_y, is_4_neighbourhood=True)
        local_coordinates = np.argwhere(neighbours > 0)
        if local_coordinates.size > 0:
            residence_coordinates = np.array([row_x, col_y]) - (center - local_coordinates)
            for residence_coordinate in residence_coordinates:
                self.road_adj_to_residence[curr_road_id].add(
                    (int(residence_coordinate[0]), int(residence_coordinate[1]))
                )

    def update_road_network(self, row_x, col_y):
        # Place/update a road tile and keep road_sets + graph consistent.
        assert int(self.grid.tile._mesa_data[(row_x, col_y)]) == TileTypes.ROAD.value

        neighbours, center = self.get_neighbours(self.road_sets, row_x, col_y, is_4_neighbourhood=True)
        road_local_coordinates = np.argwhere(neighbours > 0)
        neighbours = set(neighbours.flatten().tolist())
        try:
            neighbours.remove(0)
        except KeyError:
            pass

        curr_road_id = None
        if len(neighbours) == 0:
            # New road network
            self.road_sets[row_x, col_y] = self.road_set_id
            self.road_graph.add_node((row_x, col_y))
            curr_road_id = self.road_set_id
            self.road_set_id += 1
        else:
            # Merge into the largest neighbouring network
            max_count = -1
            max_neighbour = -1
            for neighbour in neighbours:
                total_in_network = (self.road_sets == neighbour).sum()
                if total_in_network > max_count:
                    max_count = total_in_network
                    max_neighbour = neighbour

            self.road_sets[row_x, col_y] = max_neighbour
            self.road_graph.add_node((row_x, col_y))

            road_coordinates = np.array([row_x, col_y]) - (center - road_local_coordinates)
            for road_coordinate in road_coordinates:
                self.road_graph.add_edge((row_x, col_y), (int(road_coordinate[0]), int(road_coordinate[1])))

            curr_road_id = max_neighbour
            # Relabel smaller networks into the largest one
            for neighbour in neighbours:
                if neighbour == max_neighbour:
                    continue
                self.road_sets[self.road_sets == neighbour] = max_neighbour
                self.road_adj_to_residence[max_neighbour] |= self.road_adj_to_residence[neighbour]
                if neighbour in self.road_adj_to_residence:
                    del self.road_adj_to_residence[neighbour]

            # Update adjacencies to functional tiles
            self.update_road_adjacencies(row_x, col_y, TileTypes.RESIDENCE.value, curr_road_id)
            self.update_road_adjacencies(row_x, col_y, TileTypes.INDUSTRY.value, curr_road_id)
            self.update_road_adjacencies(row_x, col_y, TileTypes.SERVICE.value, curr_road_id)

            # Cache connectivity components
            self.road_connected_sets = [node_set for node_set in nx.connected_components(self.road_graph)]

    @staticmethod
    def get_neighbours(tiles: np.array, center_row: int, center_col: int, is_4_neighbourhood: bool = False):
        """
        Returns (neighbourhood, center_index) around (center_row, center_col).
        Neighbourhood excludes center cell. With is_4_neighbourhood=True, diagonals are zeroed.
        """
        width, height = tiles.shape
        x_lower = center_row - 1
        x_upper = center_row + 2
        y_lower = center_col - 1
        y_upper = center_col + 2
        center = [1, 1]

        if x_lower < 0:
            x_lower = 0
            center[0] = 0
        if x_upper >= width:
            x_upper = width
            center[0] = 1
        if y_lower < 0:
            y_lower = 0
            center[1] = 0
        if y_upper >= height:
            y_upper = height
            center[1] = 1

        neighbours = np.copy(tiles[x_lower:x_upper, y_lower:y_upper])
        neighbours[tuple(center)] = 0

        if is_4_neighbourhood:
            diagonal_adjacents = [[], []]
            for r in [-1, 1]:
                for c in [-1, 1]:
                    r_i = center[0] + r
                    c_i = center[1] + c
                    if 0 <= r_i < neighbours.shape[0] and 0 <= c_i < neighbours.shape[1]:
                        diagonal_adjacents[0].append(r_i)
                        diagonal_adjacents[1].append(c_i)
            neighbours[diagonal_adjacents[0], diagonal_adjacents[1]] = 0

        return neighbours, np.array(center)

    def remove_road_tile(self, row_x, col_y):
        old_road_id = int(self.road_sets[row_x, col_y])
        self.road_graph.remove_node((row_x, col_y))
        new_road_connected_sets = [node_set for node_set in nx.connected_components(self.road_graph)]

        relabel_coordinates = {}
        original_set = None

        for old_set in self.road_connected_sets:
            if (row_x, col_y) in old_set:
                original_set = old_set
                break

        new_sets = []
        for new_set in new_road_connected_sets:
            if original_set is not None and new_set < original_set:
                new_sets.append(new_set)

        new_sets.sort(key=lambda x: len(x), reverse=True)
        for i, new_set in enumerate(new_sets):
            new_set_row = []
            new_set_col = []
            for x, y in new_set:
                new_set_row.append(x)
                new_set_col.append(y)
            if i == 0:
                relabel_coordinates[old_road_id] = (new_set_row, new_set_col)
            else:
                relabel_coordinates[self.road_set_id] = (new_set_row, new_set_col)
                self.road_set_id += 1

        for new_road_id, (rows, cols) in relabel_coordinates.items():
            if new_road_id == old_road_id:
                continue
            self.road_sets[rows, cols] = new_road_id

        self.road_sets[row_x, col_y] = 0
        self.road_connected_sets = new_road_connected_sets

        self.relabel_adjacencies(old_road_id, relabel_coordinates, self.residence_tiles, self.road_adj_to_residence)
        self.relabel_adjacencies(old_road_id, relabel_coordinates, self.industry_tiles, self.road_adj_to_industries)
        self.relabel_adjacencies(old_road_id, relabel_coordinates, self.service_tiles, self.road_adj_to_services)

    def relabel_adjacencies(self, old_road_id, relabel_coordinates, tiles_of_interest, adj_list):
        if old_road_id in adj_list:
            del adj_list[old_road_id]

        for new_road_id, (rows, cols) in relabel_coordinates.items():
            mask = np.zeros(self.road_sets.shape)
            mask[rows, cols] = 1
            st = generate_binary_structure(2, 1)
            mask = grey_dilation(mask, footprint=iterate_structure(st, 1), mode="constant")
            tiles_affected = tiles_of_interest * mask

            tiles_coordinates = np.argwhere(tiles_affected > 0)
            if len(tiles_coordinates) > 0:
                adj_list[new_road_id] = set([])
            for coordinate in tiles_coordinates:
                adj_list[new_road_id].add((int(coordinate[0]), int(coordinate[1])))

    # ---------------- RL hooks ----------------

    def get_observation(self):
        """
        Numeric observation for RL.
        Returns (H, W, 3): [tile_norm, pop_gain, poll_gain]
        """
        tile = self.grid.tile._mesa_data
        popg = self.grid.pop_g._mesa_data
        poll = self.grid.poll_g._mesa_data
        try:
            max_tile = float(max(t.value for t in TileTypes))
        except Exception:
            max_tile = 5.0
        denom = max_tile if max_tile > 0 else 1.0
        tile_norm = tile.astype(np.float32) / denom
        popg_f = popg.astype(np.float32)
        poll_f = poll.astype(np.float32)
        return np.stack([tile_norm, popg_f, poll_f], axis=-1)

    def compute_reward(self, action=None) -> float:
        """
        Dense reward: encourage population gain, penalize pollution gain.
        Tune LAMBDA_POLL as needed.
        """
        LAMBDA_POLL = 0.5
        pop_g = float(getattr(self.update_rules, "curr_pop_g", 0.0))
        poll_g = float(getattr(self.update_rules, "curr_poll_g", 0.0))
        return pop_g - LAMBDA_POLL * poll_g

    def is_done(self) -> bool:
        """
        Episode termination. External trainer typically enforces max steps,
        so keep False unless you want custom early stopping.
        """
        return False

    def valid_actions(self):
        """
        Iterable of valid action indices.
        Encoding: action ∈ [0, N_tiles * width * height)
          tile_idx = action // (width*height)
          cell_idx = action %  (width*height)
          x = cell_idx // self.height  (row index)
          y = cell_idx %  self.height  (col index)
        """
        return range(len(self.ACTION_TILE_TYPES) * self.width * self.height)

    def _decode_action(self, action: int):
        """
        Map flat action index -> (tile_value, x, y).
        """
        n_cells = self.width * self.height
        tile_idx = action // n_cells
        cell_idx = action % n_cells

        # Arrays are shaped (width, height); we use (row_x, col_y)
        x = cell_idx // self.height  # row index in [0, width-1]
        y = cell_idx %  self.height  # col index in [0, height-1]

        if not (0 <= tile_idx < len(self.ACTION_TILE_TYPES)):
            raise ValueError(f"tile_idx out of range: {tile_idx}")
        tile_value = self.ACTION_TILE_TYPES[tile_idx]
        return int(tile_value), int(x), int(y)

    def apply_action(self, action: int):
        """
        Execute one RL action and advance the simulation by one tick,
        mirroring what .step() does (but without planner 'decide').
        This ensures observations and reward reflect the action.
        """
        tile_value, x, y = self._decode_action(int(action))

        # 1) Mutate the grid per action
        self.set_tile(x, y, tile_value)

        # 2) Book-keeping and environment update (similar to step())
        self.book_keep()
        self.update_rules.apply_rules(self)
        self._safe_agent_update()

        # 3) Optional: collect data at the same cadence
        if self.time_step % self.collect_rate == 0:
            self.data_collectors.collect(self)

        # 4) Advance time
        self.time_step += 1
