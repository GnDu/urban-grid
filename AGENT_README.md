# BalancedCityPlanner Agent

A strategic rule-based agent for building balanced cities in the Urban Grid environment.

## Overview

The `BalancedCityPlanner` is designed to create balanced cities with configurable tile ratios while respecting strict road connectivity constraints. Unlike random placement, this agent uses a phase-based strategy to build functional, connected cities.

## Strategy

The agent operates in three phases:

1. **Road Expansion** (0-70% of target roads)
   - Builds road network to provide connectivity
   - Can be aggressive (grid-aligned) or conservative (random expansion)
   - Expands until 70% of target road ratio is achieved

2. **Zone Placement** (70-80% of functional zones)
   - Places residence, industry, and service tiles
   - Prioritizes zones with the largest deficit from target ratios
   - Creates clusters of similar tiles when zoning is enabled
   - All zones must be adjacent to roads

3. **Balancing** (Final phase)
   - Fills remaining space to reach exact target ratios
   - Adds greenery proactively if pollution exceeds threshold
   - Fine-tunes city layout to match desired distribution

## Key Features

- **Respects Road Constraints**: All tiles must connect to the road network
- **Configurable Ratios**: Adjust target percentages for each tile type
- **Strategic Placement**: Uses scoring to prefer optimal positions
- **Clustering Support**: Groups similar tiles together for efficiency
- **Pollution Control**: Proactively adds greenery when pollution is high
- **Phase Transitions**: Automatically adapts strategy as city develops

## Hyperparameters

### Tile Ratio Targets
- `target_road_ratio` (default: 0.15) - Target percentage of roads
- `target_residence_ratio` (default: 0.20) - Target percentage of residences
- `target_industry_ratio` (default: 0.20) - Target percentage of industry
- `target_service_ratio` (default: 0.20) - Target percentage of services
- `target_greenery_ratio` (default: 0.25) - Target percentage of greenery

### Placement Strategy
- `road_expansion_aggressive` (default: True) - Expand roads quickly along grid lines vs randomly
- `road_spacing` (default: 4) - Grid spacing for road network (4 = roads every 4 tiles)
- `cluster_size` (default: 3) - Preferred size for tile clusters (currently informational)
- `use_zoning` (default: True) - Group similar tiles together vs random placement

### Balancing Parameters
- `prioritize_pollution_control` (default: True) - Add greenery when pollution exceeds threshold
- `pollution_threshold` (default: 100) - Max pollution before prioritizing greenery
- `population_weight` (default: 1.0) - How much to value population gain (future use)

## Usage

### Quick Test

```python
python test_balanced_agent.py
```

This will:
- Run a 300-step simulation on a 20x20 grid
- Use default hyperparameters
- Generate visualizations (HTML animation + PNG charts)
- Print detailed statistics

### Custom Configuration

Edit the hyperparameters at the top of `test_balanced_agent.py`:

```python
# Example: More conservative, greenery-focused city
TARGET_ROAD_RATIO = 0.12           # Fewer roads
TARGET_GREENERY_RATIO = 0.35       # More greenery
ROAD_EXPANSION_AGGRESSIVE = False  # Slower road expansion
POLLUTION_THRESHOLD = 50           # Lower pollution tolerance
```

### Programmatic Use

```python
from environments.environment_stricter_road import CityModelStricterRoad
from agents.agent import BalancedCityPlanner
from update_rules.update_rules_stricter_road import UpdateRules

# Initialize environment
update_rules = UpdateRules()
init_road_tile = CityModelStricterRoad.get_random_init_road_tile(20, 20, seed=42)

model = CityModelStricterRoad(
    agent_class=BalancedCityPlanner,
    width=20,
    height=20,
    update_rules=update_rules,
    init_road_tile=init_road_tile,
    seed=42
)

# Configure agent hyperparameters
agent = model.schedule.agents[0]
agent.target_road_ratio = 0.15
agent.target_greenery_ratio = 0.30
agent.use_zoning = True

# Run simulation
for step in range(300):
    model.step()

# Access results
print(f"Total population: {agent.total_population}")
print(f"Total pollution: {agent.total_pollution}")
```

## Output Files

When running `test_balanced_agent.py`, the following files are generated:

1. **balanced_city_animation.html** - Interactive animation with playback controls
2. **balanced_city_metrics.png** - Performance charts (population, pollution, tile counts)
3. **balanced_city_final.png** - Final city layout visualization

## Tuning Tips

### Maximize Population
- Increase `target_residence_ratio` and `target_service_ratio`
- Decrease `target_greenery_ratio`
- Set `prioritize_pollution_control = False`

### Minimize Pollution
- Increase `target_greenery_ratio`
- Lower `pollution_threshold`
- Enable `prioritize_pollution_control = True`

### Efficient Road Networks
- Set `road_expansion_aggressive = True`
- Use higher `road_spacing` (5-6) for grid-like roads
- Lower `target_road_ratio` (0.10-0.12)

### Clustered Zones
- Set `use_zoning = True`
- Results in industrial districts, residential neighborhoods, etc.

### Mixed Development
- Set `use_zoning = False`
- Results in mixed-use neighborhoods

## Comparison to Other Agents

| Agent | Strategy | Training Required | Respects Constraints | Balanced Output |
|-------|----------|-------------------|---------------------|-----------------|
| RandomPlanner | Random placement | No | No | No |
| BalancedCityPlanner | Rule-based phases | No | Yes | Yes |
| RL Agent (future) | Learned policy | Yes | Yes | Potentially better |

## Evaluation Metrics

The agent is evaluated on:

1. **Target Achievement**: How close actual ratios are to target ratios
2. **Population**: Total population generated
3. **Pollution Control**: Population/Pollution ratio
4. **Connectivity**: Percentage of tiles connected to road network
5. **Space Efficiency**: Percentage of grid utilized

## Limitations

- **Rule-based**: May not find optimal solutions that RL could discover
- **No look-ahead**: Makes greedy decisions without planning future moves
- **Fixed phases**: Phase transitions are based on simple thresholds
- **No learning**: Doesn't improve from experience across episodes

## Future Improvements

- Add genetic algorithm for hyperparameter optimization
- Implement multi-objective optimization (population vs pollution)
- Add pattern recognition for copying successful layouts
- Support warm_start() for transfer learning
- Dynamic phase transitions based on metrics instead of ratios

## License

Part of the Urban Grid project.
