import mesa
from utils import TileTypes

class CityPlanner(mesa.Agent):
    """
    Abstract class for agent in SIM city 
    """

    def __init__(self, model, destroy_tile_penalty:float=10, **kwargs):
        super().__init__(model)
        self.destroy_tile_penalty = destroy_tile_penalty
        self.total_population = 0
        self.total_pollution = 0
        self.population_cap = 0

    def decide(self):
        """
        How the agent decide what step to take next
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")


    def update(self, **kwargs):
        """
        How the agent update any internal state
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")
    
    def warm_start(self, **kwargs):
        """
        Initializing a new agent with knowledge, policies, or parameters learned from previous episodes or tasks
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")
    
    #the two actions

    def place(self, row_x, col_y, tile):
        #check if tile is even applicable. Throw an error if that is the case
        #should check for any illegal action before hand
        #note, tile should not be BARREN. That's destroy tile
        assert(tile!=TileTypes.BARREN.value)
        x_y_tile = self.model.get_tile(row_x, col_y)
        if x_y_tile!=TileTypes.BARREN.value:
            raise RuntimeError(f"({row_x}, {col_y}): {x_y_tile} is not BARREN")
        self.model.set_tile(row_x, col_y, tile)
        
    def destroy(self, row_x, col_y):
        #revert the tile to a barren
        #if x,y is not barren, add a fix poll_g to total_pollution
        self.model.set_tile(row_x, col_y, TileTypes.BARREN.value)
        #increment total polution
        self.total_pollution+=self.destroy_tile_penalty
    

class RandomPlanner(CityPlanner):

    def __init__(self, model):
        super().__init__(model)

    def decide(self):
        #just randomly pick a random tile and goooo
        self.model.width
        self.model.height


class BalancedCityPlanner(CityPlanner):
    """
    Strategic agent that builds a balanced city with configurable tile ratios.

    Strategy:
    1. Expand road network intelligently to provide connectivity
    2. Place functional zones (residence, industry, service) in clusters
    3. Add greenery to balance pollution
    4. Adapt based on population and pollution metrics

    Hyperparameters allow tuning tile ratios and placement strategies.
    """

    def __init__(self, model,
                 # Tile ratio targets (as percentages of total grid)
                 target_road_ratio=0.15,
                 target_residence_ratio=0.20,
                 target_industry_ratio=0.20,
                 target_service_ratio=0.20,
                 target_greenery_ratio=0.25,

                 # Placement strategy parameters
                 road_expansion_aggressive=True,  # Expand roads quickly vs conservatively
                 cluster_size=3,  # Preferred size for tile clusters (3x3, etc)
                 prioritize_pollution_control=True,  # Add greenery proactively

                 # Balancing parameters
                 population_weight=1.0,  # How much to value population vs pollution
                 pollution_threshold=100,  # Max pollution before prioritizing greenery

                 # Placement patterns
                 use_zoning=True,  # Group similar tiles together vs mix
                 road_spacing=4,  # Grid spacing for road network

                 **kwargs):
        super().__init__(model, **kwargs)

        # Store hyperparameters
        self.target_road_ratio = target_road_ratio
        self.target_residence_ratio = target_residence_ratio
        self.target_industry_ratio = target_industry_ratio
        self.target_service_ratio = target_service_ratio
        self.target_greenery_ratio = target_greenery_ratio

        self.road_expansion_aggressive = road_expansion_aggressive
        self.cluster_size = cluster_size
        self.prioritize_pollution_control = prioritize_pollution_control

        self.population_weight = population_weight
        self.pollution_threshold = pollution_threshold

        self.use_zoning = use_zoning
        self.road_spacing = road_spacing

        # Internal state
        self.step_count = 0
        self.phase = "road_expansion"  # Phases: road_expansion, zone_placement, balancing
        self.zone_queue = []  # Queue of zones to place

    def decide(self):
        """
        Strategic decision-making based on current city state and phase.
        """
        import numpy as np

        self.step_count += 1

        # Get current tile counts
        total_tiles = self.model.width * self.model.height
        road_count = np.sum(self.model.road_tiles)
        residence_count = np.sum(self.model.residence_tiles)
        industry_count = np.sum(self.model.industry_tiles)
        service_count = np.sum(self.model.service_tiles)
        greenery_count = np.sum(self.model.greenery_tiles)
        barren_count = np.sum(self.model.grid.tile._mesa_data == TileTypes.BARREN.value)

        # Calculate current ratios
        road_ratio = road_count / total_tiles
        residence_ratio = residence_count / total_tiles
        industry_ratio = industry_count / total_tiles
        service_ratio = service_count / total_tiles
        greenery_ratio = greenery_count / total_tiles

        # Phase transitions based on ratios
        if self.phase == "road_expansion":
            # Stay in road expansion longer to ensure better connectivity
            # Transition only when we have enough roads AND good network coverage
            if road_ratio >= self.target_road_ratio * 0.9:  # 90% of target roads built
                self.phase = "zone_placement"

        if self.phase == "zone_placement":
            # Transition to balancing if targets are met OR after enough steps
            # This prevents getting stuck if targets are unreachable
            if ((residence_ratio >= self.target_residence_ratio * 0.8 and
                industry_ratio >= self.target_industry_ratio * 0.8 and
                service_ratio >= self.target_service_ratio * 0.8) or
                self.step_count > 250):  # Fallback after 250 steps
                self.phase = "balancing"

        # Decision logic based on phase
        try:
            if self.phase == "road_expansion":
                self._place_road_strategic()
            elif self.phase == "zone_placement":
                # In zone placement, check if we need more roads to reach barren areas
                # If there are many barren tiles and few legal placements near roads, expand roads
                legal_roads = self.model.get_legal_road_tiles()
                from scipy.ndimage import grey_dilation, iterate_structure, generate_binary_structure
                st = generate_binary_structure(2, 1)
                tiles_near_roads = grey_dilation(self.model.road_tiles,
                                                footprint=iterate_structure(st, 1),
                                                mode='constant')
                tiles_near_roads -= self.model.road_tiles
                valid_tiles_near_roads = tiles_near_roads * (self.model.grid.tile._mesa_data == TileTypes.BARREN.value)
                reachable_barren = np.sum(valid_tiles_near_roads > 0)

                # Expand roads if significant barren areas are unreachable
                # Goal: reach ALL areas, not just meet quotas
                should_expand = (
                    barren_count > total_tiles * 0.2 and  # More than 20% barren
                    reachable_barren < barren_count * 0.5 and  # Less than 50% reachable
                    len(legal_roads) > 0
                )

                if should_expand:
                    self._place_road_strategic()
                else:
                    self._place_functional_zone(residence_ratio, industry_ratio, service_ratio)
            else:  # balancing phase
                # Keep filling until grid is essentially full
                # Don't stop based on ratio - try to fill everything possible
                self._balance_city(road_ratio, residence_ratio, industry_ratio,
                                  service_ratio, greenery_ratio, barren_count)
        except (RuntimeError, AssertionError) as e:
            # If placement fails, try alternative action
            self._place_any_available()

    def _place_road_strategic(self):
        """
        Expand road network strategically.
        """
        import numpy as np

        legal_road_tiles = self.model.get_legal_road_tiles()

        if len(legal_road_tiles) == 0:
            return  # No valid road placements

        if self.road_expansion_aggressive:
            # Expand roads along main axes to create grid network
            # Prefer tiles that are aligned with road_spacing intervals
            scores = []
            for row, col in legal_road_tiles:
                score = 0
                # Prefer grid-aligned positions
                if row % self.road_spacing == 0 or col % self.road_spacing == 0:
                    score += 10
                # Prefer positions that extend the network
                score += self._count_barren_neighbors(row, col)
                scores.append(score)

            best_idx = np.argmax(scores)
            row, col = legal_road_tiles[best_idx]
        else:
            # Conservative: random valid road placement
            idx = np.random.randint(len(legal_road_tiles))
            row, col = legal_road_tiles[idx]

        self.place(row, col, TileTypes.ROAD.value)

    def _place_functional_zone(self, residence_ratio, industry_ratio, service_ratio):
        """
        Place residence, industry, or service tiles based on current ratios.
        """
        import numpy as np

        # Determine which tile type needs more placement
        residence_need = self.target_residence_ratio - residence_ratio
        industry_need = self.target_industry_ratio - industry_ratio
        service_need = self.target_service_ratio - service_ratio

        # Choose tile type with highest need
        needs = [
            (residence_need, TileTypes.RESIDENCE.value, "residence"),
            (industry_need, TileTypes.INDUSTRY.value, "industry"),
            (service_need, TileTypes.SERVICE.value, "service")
        ]
        needs.sort(reverse=True)

        for need, tile_type, name in needs:
            if need > 0:
                if self._place_tile_near_roads(tile_type):
                    return

        # If all functional zones are satisfied, place greenery
        self._place_tile_near_roads(TileTypes.GREENERY.value)

    def _balance_city(self, road_ratio, residence_ratio, industry_ratio,
                     service_ratio, greenery_ratio, barren_count):
        """
        Final balancing phase: fill remaining space optimally.
        """
        import numpy as np
        from scipy.ndimage import grey_dilation, iterate_structure, generate_binary_structure

        # Calculate grid size early for use in conditions
        total_tiles = self.model.width * self.model.height

        # Check if we need pollution control
        # Only apply early pollution control if map is mostly filled (barren < 10%)
        # This prevents pollution control from interfering with initial filling
        if (self.prioritize_pollution_control and
            self.total_pollution > self.pollution_threshold and
            barren_count < total_tiles * 0.1):  # Only if mostly filled
            if self._place_tile_near_roads(TileTypes.GREENERY.value):
                return

        # Priority 1: ALWAYS expand roads to reach unreachable barren areas
        # Goal: ZERO barren tiles - coverage is more important than target ratios
        legal_roads = self.model.get_legal_road_tiles()

        if barren_count > 0:
            # Check how many barren tiles are reachable (adjacent to roads)
            st = generate_binary_structure(2, 1)
            tiles_near_roads = grey_dilation(self.model.road_tiles,
                                            footprint=iterate_structure(st, 1),
                                            mode='constant')
            tiles_near_roads -= self.model.road_tiles
            valid_tiles_near_roads = tiles_near_roads * (self.model.grid.tile._mesa_data == TileTypes.BARREN.value)
            reachable_barren = np.sum(valid_tiles_near_roads > 0)

            # If less than 50% of barren is reachable, expand roads FIRST to reach them
            # Lowered from 60% to further reduce excessive road placement
            if reachable_barren < barren_count * 0.5 and len(legal_roads) > 0:
                scores = []
                for row, col in legal_roads:
                    score = 0
                    # Heavily prioritize positions that border more barren tiles
                    score += self._count_barren_neighbors(row, col) * 10
                    # Add distance-based score to reach far areas
                    score += self._get_expansion_score(row, col)
                    scores.append(score)

                best_idx = np.argmax(scores)
                row, col = legal_roads[best_idx]
                self.place(row, col, TileTypes.ROAD.value)
                return

            # If most barren is reachable (>70%), FILL IT with functional zones
            # Don't just keep placing roads - fill the reachable spaces!
            if reachable_barren > 0:
                # Try to place a functional zone in reachable barren area
                # Prioritize by needs, but ensure we're actually FILLING not just expanding
                pass  # Fall through to needs calculation below

        # Priority 2: Calculate needs for all tile types
        # If barren remains, ALWAYS keep filling even if targets met
        # But heavily prioritize functional zones over roads when areas are reachable
        if barren_count > 0:
            # Calculate road excess - if we have way too many roads, strongly avoid more
            road_excess = max(0, road_ratio - self.target_road_ratio)
            road_penalty = road_excess * 2  # Penalize roads if we're over target

            # SMART POLLUTION CONTROL: If population is at cap, stop placing polluting zones
            # and prioritize greenery + residences to increase cap
            population_at_cap = (self.population_cap > 0 and
                                self.total_population >= self.population_cap * 0.95)

            if population_at_cap:
                # Population capped! Prioritize residences (to raise cap) and greenery (to reduce pollution)
                # Minimize industry/service to avoid runaway pollution
                needs = {
                    TileTypes.ROAD.value: max(0.01 - road_penalty, 0.01, self.target_road_ratio - road_ratio),
                    TileTypes.RESIDENCE.value: max(0.7, self.target_residence_ratio - residence_ratio),  # Highest priority - raise cap!
                    TileTypes.INDUSTRY.value: 0.05,  # Minimal - stop polluting!
                    TileTypes.SERVICE.value: 0.05,   # Minimal - stop polluting!
                    TileTypes.GREENERY.value: max(0.6, self.target_greenery_ratio - greenery_ratio),  # High - control pollution!
                }
            else:
                # Normal balanced growth - prioritize residences for population growth
                needs = {
                    TileTypes.ROAD.value: max(0.01 - road_penalty, 0.01, self.target_road_ratio - road_ratio),
                    TileTypes.RESIDENCE.value: max(0.5, self.target_residence_ratio - residence_ratio),  # Higher priority
                    TileTypes.INDUSTRY.value: max(0.25, self.target_industry_ratio - industry_ratio),
                    TileTypes.SERVICE.value: max(0.25, self.target_service_ratio - service_ratio),
                    TileTypes.GREENERY.value: max(0.35, self.target_greenery_ratio - greenery_ratio),
                }
        else:
            needs = {
                TileTypes.ROAD.value: max(0, self.target_road_ratio - road_ratio),
                TileTypes.RESIDENCE.value: max(0, self.target_residence_ratio - residence_ratio),
                TileTypes.INDUSTRY.value: max(0, self.target_industry_ratio - industry_ratio),
                TileTypes.SERVICE.value: max(0, self.target_service_ratio - service_ratio),
                TileTypes.GREENERY.value: max(0, self.target_greenery_ratio - greenery_ratio),
            }

        # If all targets met, KEEP FILLING with balanced tiles until grid is full
        total_need = sum(needs.values())
        if total_need <= 0.05 and barren_count > 10:  # Targets met but barren remains
            # Fill remaining space proportionally - NO MORE ROADS, prioritize residences and greenery
            needs = {
                TileTypes.GREENERY.value: 0.35,
                TileTypes.RESIDENCE.value: 0.35,  # Higher priority for population growth
                TileTypes.SERVICE.value: 0.15,
                TileTypes.INDUSTRY.value: 0.19,
                TileTypes.ROAD.value: 0.01,  # Absolute minimum roads, focus on filling
            }

        # Sort by need
        sorted_needs = sorted(needs.items(), key=lambda x: x[1], reverse=True)

        # Try to place most needed tile type
        placed_something = False
        for tile_type, need in sorted_needs:
            if need > 0:
                if tile_type == TileTypes.ROAD.value:
                    # Use strategic road placement instead of random
                    try:
                        self._place_road_strategic()
                        placed_something = True
                        return
                    except:
                        pass
                else:
                    if self._place_tile_near_roads(tile_type):
                        placed_something = True
                        return

        # If nothing was placed, try ALL tile types near roads (ignore needs)
        if not placed_something and barren_count > 0:
            for tile_type in [TileTypes.GREENERY.value, TileTypes.RESIDENCE.value,
                             TileTypes.INDUSTRY.value, TileTypes.SERVICE.value]:
                if self._place_tile_near_roads(tile_type):
                    placed_something = True
                    return

        # If still nothing placed, try expanding roads to reach barren areas
        if not placed_something and barren_count > 0 and len(legal_roads) > 0:
            # Use strategic placement toward barren areas
            scores = []
            for row, col in legal_roads:
                score = 0
                score += self._count_barren_neighbors(row, col) * 10
                score += self._get_expansion_score(row, col)
                scores.append(score)

            best_idx = np.argmax(scores)
            row, col = legal_roads[best_idx]
            try:
                self.place(row, col, TileTypes.ROAD.value)
                placed_something = True
                return
            except:
                pass

        # Last resort: try any available placement
        if not placed_something:
            self._place_any_available()

    def _place_tile_near_roads(self, tile_type):
        """
        Find valid placement for non-road tile adjacent to road network.
        Returns True if successful placement, False otherwise.
        """
        import numpy as np
        from scipy.ndimage import grey_dilation, iterate_structure, generate_binary_structure

        # Find barren tiles adjacent to roads
        st = generate_binary_structure(2, 1)
        tiles_near_roads = grey_dilation(self.model.road_tiles,
                                        footprint=iterate_structure(st, 1),
                                        mode='constant')
        tiles_near_roads -= self.model.road_tiles

        # Only barren tiles are valid
        valid_tiles = tiles_near_roads * (self.model.grid.tile._mesa_data == TileTypes.BARREN.value)
        valid_positions = np.argwhere(valid_tiles > 0)

        if len(valid_positions) == 0:
            return False

        if self.use_zoning:
            # Prefer positions near same tile type (clustering)
            # Also prioritize filling isolated barren holes
            scores = []
            for row, col in valid_positions:
                score = self._count_same_type_neighbors(row, col, tile_type)

                # IMPORTANT: Bonus for positions that fill "holes" (barren tiles surrounded by non-barren)
                # Count how many non-barren neighbors this position has
                non_barren_neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.model.height and 0 <= nc < self.model.width:
                        if self.model.grid.tile._mesa_data[nr, nc] != TileTypes.BARREN.value:
                            non_barren_neighbors += 1

                # If mostly surrounded by non-barren tiles, this is a "hole" - prioritize it
                if non_barren_neighbors >= 3:
                    score += 20  # High bonus for filling holes

                scores.append(score)

            best_idx = np.argmax(scores)
            row, col = valid_positions[best_idx]
        else:
            # Random valid position
            idx = np.random.randint(len(valid_positions))
            row, col = valid_positions[idx]

        # Before placing, check if this would isolate any barren neighbors
        # If placing here would trap barren tiles with no road access, skip it
        if self._would_isolate_barren(row, col):
            # Try a different position - don't trap barren tiles!
            # Find alternative positions that don't isolate barren
            for alt_row, alt_col in valid_positions:
                if (alt_row, alt_col) != (row, col) and not self._would_isolate_barren(alt_row, alt_col):
                    try:
                        self.place(alt_row, alt_col, tile_type)
                        return True
                    except (RuntimeError, AssertionError):
                        continue
            # If all positions would isolate barren, place anyway (better than nothing)
            pass

        try:
            self.place(row, col, tile_type)
            return True
        except (RuntimeError, AssertionError):
            return False

    def _place_any_available(self):
        """
        Fallback: place any valid tile when strategic placement fails.
        """
        import numpy as np

        # Try placing road toward isolated barren holes first
        legal_roads = self.model.get_legal_road_tiles()
        if len(legal_roads) > 0:
            # Score roads by proximity to barren holes
            scores = []
            for row, col in legal_roads:
                score = 0
                # Check if this road would help reach barren tiles surrounded by non-barren
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.model.height and 0 <= nc < self.model.width and
                        self.model.grid.tile._mesa_data[nr, nc] == TileTypes.BARREN.value):
                        # Check if this barren tile is surrounded (a hole)
                        surrounding_non_barren = 0
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr2, nc2 = nr + dr2, nc + dc2
                            if 0 <= nr2 < self.model.height and 0 <= nc2 < self.model.width:
                                if self.model.grid.tile._mesa_data[nr2, nc2] != TileTypes.BARREN.value:
                                    surrounding_non_barren += 1
                        if surrounding_non_barren >= 2:  # Adjacent to a potential hole
                            score += 10

                # Also prefer positions with more barren neighbors (general expansion)
                score += self._count_barren_neighbors(row, col)
                scores.append(score)

            best_idx = np.argmax(scores)
            row, col = legal_roads[best_idx]
            try:
                self.place(row, col, TileTypes.ROAD.value)
                return
            except:
                pass

        # Try any non-road tile near roads
        for tile_type in [TileTypes.GREENERY.value, TileTypes.RESIDENCE.value,
                         TileTypes.SERVICE.value, TileTypes.INDUSTRY.value]:
            if self._place_tile_near_roads(tile_type):
                return

    def _count_barren_neighbors(self, row, col):
        """
        Count how many barren tiles are adjacent to this position.
        """
        import numpy as np
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.model.height and 0 <= nc < self.model.width:
                if self.model.grid.tile._mesa_data[nr, nc] == TileTypes.BARREN.value:
                    count += 1
        return count

    def _count_same_type_neighbors(self, row, col, tile_type):
        """
        Count how many tiles of the same type are adjacent (for clustering).
        """
        import numpy as np
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.model.height and 0 <= nc < self.model.width:
                if self.model.grid.tile._mesa_data[nr, nc] == tile_type:
                    count += 1
        return count

    def _would_isolate_barren(self, row, col):
        """
        Check if placing a functional zone at (row, col) would isolate any barren neighbors
        from road access. Returns True if it would trap barren tiles.
        """
        import numpy as np

        # Check each adjacent barren tile
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.model.height and 0 <= nc < self.model.width and
                self.model.grid.tile._mesa_data[nr, nc] == TileTypes.BARREN.value):

                # This neighbor is barren - check if it would be isolated after our placement
                # Count how many road neighbors it would have after we place our tile
                road_neighbors = 0
                for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr2, nc2 = nr + dr2, nc + dc2
                    if 0 <= nr2 < self.model.height and 0 <= nc2 < self.model.width:
                        # Skip the position we're about to fill
                        if (nr2, nc2) == (row, col):
                            continue
                        if self.model.road_tiles[nr2, nc2] > 0:
                            road_neighbors += 1

                # If this barren tile would have NO road neighbors after our placement, we'd isolate it
                if road_neighbors == 0:
                    return True

        return False

    def _get_expansion_score(self, row, col):
        """
        Calculate a score for road expansion toward unexplored/barren areas.
        Higher scores for positions that help reach distant barren regions.
        """
        import numpy as np

        score = 0

        # 1. Distance from center of mass of existing roads
        # This encourages expansion in all directions
        road_positions = np.argwhere(self.model.road_tiles > 0)
        if len(road_positions) > 0:
            center_y, center_x = road_positions.mean(axis=0)
            # Distance from center (farther = higher score = more exploration)
            distance = np.sqrt((row - center_y)**2 + (col - center_x)**2)
            score += distance

        # 2. Bonus for positions near unexplored edges/corners
        # Check distance to grid boundaries
        dist_to_edges = min(row, self.model.height - 1 - row,
                           col, self.model.width - 1 - col)

        # Count how many barren tiles are in the direction of this position
        # from the road network center
        if len(road_positions) > 0:
            # Direction vector from center to this position
            center_y, center_x = road_positions.mean(axis=0)
            if abs(row - center_y) > 0.1 or abs(col - center_x) > 0.1:
                # Look ahead in this direction for barren tiles
                direction_y = np.sign(row - center_y) if abs(row - center_y) > 0.1 else 0
                direction_x = np.sign(col - center_x) if abs(col - center_x) > 0.1 else 0

                # Count barren tiles in a small cone ahead
                barren_ahead = 0
                for dist in range(1, 5):  # Look 4 tiles ahead
                    check_y = int(row + direction_y * dist)
                    check_x = int(col + direction_x * dist)
                    if (0 <= check_y < self.model.height and
                        0 <= check_x < self.model.width):
                        if self.model.grid.tile._mesa_data[check_y, check_x] == TileTypes.BARREN.value:
                            barren_ahead += 1
                score += barren_ahead * 2  # Bonus for reaching toward barren areas

        return score

    def update(self, **kwargs):
        """
        Update internal state after environment step.
        """
        # Update cumulative metrics
        # Access from update_rules object, not model
        self.total_population += self.model.update_rules.curr_pop_g

        # Pollution is STATE-BASED, not cumulative - reflects current tile configuration
        self.total_pollution = self.model.update_rules.curr_poll_g

        self.population_cap = self.model.update_rules.population_cap

    def warm_start(self, **kwargs):
        """
        Initialize with previous knowledge (not implemented for rule-based agent).
        """
        pass
