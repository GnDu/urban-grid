# City Simulation Update Rules Breakdown

## Overview

This simulation models a city as a grid where each tile can be one of five types: **Residence**, **Greenery** (barren/parks), **Industry**, **Service**, or **Road**. Each step, the system calculates two key metrics across the entire grid:

- **Population Generation (pop_g)**: How much population each tile attracts or supports
- **Pollution Generation (poll_g)**: How much pollution each tile produces or mitigates

## Game Loop

You can find this in `environment.py`

```python
def step(self):
    
    self.agents.do("decide")
    self.book_keep()
    #update the environment based on agent decision
    self.update_rules.apply_rules(self)
    #update any internal states, like utiity, etc
    self.agents[0].update()
    #collect the data
    if self.time_step%self.collect_rate==0:
        self.data_collectors.collect(self)
    self.time_step+=1
```

**Steps:**

1) `self.agents.do("decide")`: Invoke 'decide' method in agent. In this case, the agent will decide to either place a tile, destroy a tile or any other actions it is defined.
2) `self.book_keep()`: Book keep basically create binary array for each tile. 
3) `self.update_rules.apply_rules(self)`: apply game rules. See Core Mechanics
4) `self.agents[0].update()`: update agent states, score or utility
5) `self.data_collectors.collect(self)`: collect the metrics and data for the timestep if it's within collect rate.
6) `self.time_step+=1`: Advance to next time step, nuff said. Do note, this is the last step.

**Duration:**

The duration of the game is arbritrarily but I am suggesting width * height + number of destroy actions. This allows the agent to fully utilise the grid while having some chances to revert any mistakes. 

## Core Mechanics

### Population Cap System

The total population capacity is determined by residence tiles:

```
population_cap = residence_population_increase × (number of residence tiles)
```

This creates a dynamic population modifier that slows growth as the city approaches capacity:

```
population_modifier = (population_cap - current_total_population) / population_cap
```

All population generation is multiplied by this modifier, so growth naturally slows as capacity is reached.

### Coverage System

Some tiles affect neighboring cells within a certain radius (using Chebyshev/chessboard distance, which includes diagonals):

- **Industry Pollution**: Spreads to neighboring cells within coverage distance
- **Greenery Mitigation**: Reduces pollution in neighboring cells within coverage distance  
- **Service Modifier**: Boosts population generation for residence tiles within coverage distance

## Tile Contributions

### Population Generation (pop_g)

| Tile Type | Base Contribution | Parameter | Special Rules |
|-----------|------------------|-----------|---------------|
| **Residence** | +1.0 per tile | `residence_pop_g` | Receives +20% bonus if within 2 cells of a service tile |
| **Greenery** | +2.0 per tile | `greenery_pop_g` | No special modifiers |
| **Industry** | +8.0 per tile | `industry_pop_g` | Only if road-connected (currently skipped in code) |
| **Service** | +4.0 per tile | `service_pop_g` | Only if road-connected (currently skipped in code) |
| **Road** | +0.5 per tile | `road_pop_g` | No special modifiers |

**Service Coverage Bonus**: Residence tiles within `service_coverage` (2 cells) of any service tile receive a `service_pop_modifier` multiplier of 1.2× (1.0 + 0.2) to their population generation.

### Pollution Generation (poll_g)

| Tile Type | Base Contribution | Parameter | Special Rules |
|-----------|------------------|-----------|---------------|
| **Residence** | +1.5 per tile | `residence_poll_g` | Direct pollution on the tile only |
| **Industry** | +7.0 per tile | `industry_poll_g` | Spreads to all cells within 2-cell radius |
| **Service** | +5.0 per tile | `service_poll_g` | Direct pollution on the tile only |
| **Road** | +1.0 per tile | `road_poll_g` | Direct pollution on the tile only |
| **Greenery** | -2.0 mitigation | `greenery_poll_minus` | Reduces pollution in all cells within 2-cell radius |

**Industry Pollution Spread**: Each industry tile creates a pollution cloud that extends `industry_coverage` (2 cells) in all directions. All affected cells receive the full `industry_poll_g` value (7.0), not divided or attenuated.

**Greenery Mitigation**: Each greenery tile reduces pollution by `greenery_poll_minus` (2.0) in all cells within `greenery_coverage` (2 cells). Multiple greenery tiles can stack their mitigation effects.

**Pollution Floor**: Pollution cannot go below zero. After greenery mitigation is applied, any negative values are set to 0.

## Step-by-Step Calculation Process

### 1. Calculate Population Cap
```
population_cap = 10 × (count of residence tiles)
```

### 2. Reset Grids
Both `pop_g` and `poll_g` grids are reset to zero at the start of each update.

### 3. Calculate Base Population Generation
- Add contribution from each tile type to the `pop_g` grid
- Apply service coverage bonus to residence tiles (multiply affected tiles by 1.2)

### 4. Calculate Base Pollution Generation
- Add direct pollution from residence, service, and road tiles
- Apply industry pollution spreading using dilation
- Subtract greenery mitigation using dilation
- Clamp all negative pollution values to 0 - this means you can never 'take back' pollution already calculated.

### 5. Aggregate Results
- **Total Population Growth**: Sum of all `pop_g` values (after modifier applied)
  - Apply Population Modifier: `population_modifier = (population_cap - current_population) / population_cap`
  - First sum of all `pop_g` then mulltiply this modifier. Do note, there might be cases where the values under/overflows, I have tried to mitigate it but if you see sudden spike to the power of ridiculous numnber, chances are this is the culprit.
- **Total Pollution**: Sum of all `poll_g` values

## Tile considerations

### Spatial Effects
- **Industry** is the most polluting tile type and affects a large area around it
- **Greenery** provides the only pollution mitigation mechanism and also has area coverage
- **Services** boost residence population generation when placed nearby

### Balance Considerations
- Industry generates high pollution but also high population. This is to simulate job opportunities.
- Greenery mitigates moderate pollution  while generating decent population. This is to simulate environmental appeal
- The service bonus (20%) encourages mixed-use development with residences near services. This is to simulate amenities, and also allow areas there to be more 'attractive'
- Residence gaining population simulates living space.
- Road gaining population simulates amenities.
- In general, the ranking in terms population gain should be this: Industry > Services > Greenery > Residence > Road​. Industry generate most population, road the least.
- For pollution, this should be: Industry > Services > Road > Residence​ > Greenery.
- Note, Barren tiles neither produce pollution or population

### Growth Dynamics
The population modifier creates a natural S-curve growth pattern:
- When population is low relative to capacity: modifier near 1.0 (fast growth)
- When population approaches capacity: modifier approaches 0 (growth slows)
- This prevents unlimited exponential growth

## Parameter Reference Table

| Parameter | Value | Used For |
|-----------|-------|----------|
| `residence_population_increase` | 10 | Population cap calculation |
| `residence_poll_g` | 1.5 | Residence pollution |
| `residence_pop_g` | 1.0 | Residence population generation |
| `greenery_poll_minus` | 2.0 | Pollution mitigation per coverage cell |
| `greenery_pop_g` | 2.0 | Greenery population generation |
| `greenery_coverage` | 2 | Mitigation radius (cells) |
| `industry_poll_g` | 7.0 | Pollution per coverage cell |
| `industry_pop_g` | 8.0 | Industry population generation |
| `industry_coverage` | 2 | Pollution spread radius (cells) |
| `service_poll_g` | 5.0 | Service pollution |
| `service_pop_g` | 4.0 | Service population generation |
| `service_pop_modifier` | 0.2 | Bonus multiplier for nearby residences (adds 20%) |
| `service_coverage` | 2 | Service effect radius (cells) |
| `road_poll_g` | 1.0 | Road pollution |
| `road_pop_g` | 0.5 | Road population generation |