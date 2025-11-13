"""
Test script for BalancedCityPlanner agent.

This script allows you to:
1. Test the BalancedCityPlanner agent with the stricter road environment
2. Adjust hyperparameters easily
3. Visualize the results (HTML animation and metrics)
4. See final statistics

Usage:
    python test_balanced_agent.py
"""

import numpy as np
import pandas as pd
import json
from environments.environment_stricter_road import CityModelStricterRoad
from agents.agent import BalancedCityPlanner
from update_rules.update_rules_stricter_road import UpdateRulesStricterRoad
from update_rules.update_rules import DefaultUpdateRulesParameters
from visualisation import render_3d_array_to_gif
from utils import TileTypes
import matplotlib.pyplot as plt


# =============================================================================
# HYPERPARAMETERS - ADJUST THESE TO TEST DIFFERENT STRATEGIES
# =============================================================================

# Grid configuration
GRID_WIDTH = 10
GRID_HEIGHT = 10
NUM_STEPS = 800  # Number of simulation steps (increased to ensure full coverage)

TARGET_ROAD_RATIO = 0.18      # 18% roads 
TARGET_RESIDENCE_RATIO = 0.30  # 30% residences 
TARGET_INDUSTRY_RATIO = 0.15   # 15% industry 
TARGET_SERVICE_RATIO = 0.12    # 12% services 
TARGET_GREENERY_RATIO = 0.23   # 23% greenery


ROAD_EXPANSION_AGGRESSIVE = True 
CLUSTER_SIZE = 5                  
PRIORITIZE_POLLUTION_CONTROL = True  

POPULATION_WEIGHT = 1.0     
POLLUTION_THRESHOLD = 100    

USE_ZONING = True              
ROAD_SPACING = 6

# Visualization settings
OUTPUT_GIF = "balanced_city_animation.gif"
PIXEL_SIZE = 10               # Size of each cell in GIF (pixels)
GIF_DURATION = 100            # Duration per frame in milliseconds

# Random seed (set to None for random, or integer for reproducibility)
RANDOM_SEED = 42

# JSON trace output
OUTPUT_TRACE_JSON = "run1_trace.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tile_type_name(tile_value):
    """Convert tile value to tile type name."""
    tile_map = {
        TileTypes.BARREN.value: "BARREN",
        TileTypes.RESIDENCE.value: "RESIDENCE",
        TileTypes.GREENERY.value: "GREENERY",
        TileTypes.INDUSTRY.value: "INDUSTRY",
        TileTypes.SERVICE.value: "SERVICE",
        TileTypes.ROAD.value: "ROAD"
    }
    return tile_map.get(tile_value, "UNKNOWN")


# =============================================================================
# RUN SIMULATION
# =============================================================================

def run_simulation():
    """Run the simulation with configured hyperparameters."""

    print("=" * 70)
    print("BALANCED CITY PLANNER - SIMULATION TEST")
    print("=" * 70)
    print(f"\nGrid size: {GRID_WIDTH}x{GRID_HEIGHT}")
    print(f"Simulation steps: {NUM_STEPS}")
    print(f"Random seed: {RANDOM_SEED}")

    print("\n--- Agent Configuration ---")
    print(f"Target ratios:")
    print(f"  Roads:      {TARGET_ROAD_RATIO:.1%}")
    print(f"  Residences: {TARGET_RESIDENCE_RATIO:.1%}")
    print(f"  Industry:   {TARGET_INDUSTRY_RATIO:.1%}")
    print(f"  Services:   {TARGET_SERVICE_RATIO:.1%}")
    print(f"  Greenery:   {TARGET_GREENERY_RATIO:.1%}")
    print(f"\nStrategy:")
    print(f"  Road expansion: {'Aggressive' if ROAD_EXPANSION_AGGRESSIVE else 'Conservative'}")
    print(f"  Road spacing: {ROAD_SPACING}")
    print(f"  Use zoning: {USE_ZONING}")
    print(f"  Prioritize pollution control: {PRIORITIZE_POLLUTION_CONTROL}")
    print(f"  Pollution threshold: {POLLUTION_THRESHOLD}")

    # Load update rules from JSON configuration file
    print("\n--- Loading Update Rules ---")
    config_path = "data/update_parameters/UpdateRule_Stricter_Road.json"
    print(f"Loading configuration from: {config_path}")

    with open(config_path) as f:
        rule_parameters_dict = json.load(f)

    # Create parameters object from JSON
    rule_parameters = DefaultUpdateRulesParameters(**rule_parameters_dict)

    # Initialize update rules and set parameters
    update_rules = UpdateRulesStricterRoad()
    update_rules.set_parameters(rule_parameters)

    print(f"Configuration loaded successfully:")
    print(f"  Residence population increase: {rule_parameters.residence_population_increase}")
    print(f"  Greenery pollution reduction: {rule_parameters.greenery_poll_minus}")
    print(f"  Industry connectivity cap: {rule_parameters.industry_connectivity_cap}")
    print(f"  Service connectivity cap: {rule_parameters.service_connectivity_cap}")

    # Get random initial road position
    init_road_tile = CityModelStricterRoad.get_random_init_road_tile(
        GRID_HEIGHT, GRID_WIDTH, seed=RANDOM_SEED
    )
    print(f"\nInitial road tile: {init_road_tile}")

    # Create model with BalancedCityPlanner agent
    model = CityModelStricterRoad(
        agent_class=BalancedCityPlanner,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        update_rules=update_rules,
        init_road_tile=init_road_tile,
        collect_rate=1.0,
        seed=RANDOM_SEED
    )

    # Configure agent with hyperparameters
    agent = model.get_city_planner()
    agent.target_road_ratio = TARGET_ROAD_RATIO
    agent.target_residence_ratio = TARGET_RESIDENCE_RATIO
    agent.target_industry_ratio = TARGET_INDUSTRY_RATIO
    agent.target_service_ratio = TARGET_SERVICE_RATIO
    agent.target_greenery_ratio = TARGET_GREENERY_RATIO
    agent.road_expansion_aggressive = ROAD_EXPANSION_AGGRESSIVE
    agent.cluster_size = CLUSTER_SIZE
    agent.prioritize_pollution_control = PRIORITIZE_POLLUTION_CONTROL
    agent.population_weight = POPULATION_WEIGHT
    agent.pollution_threshold = POLLUTION_THRESHOLD
    agent.use_zoning = USE_ZONING
    agent.road_spacing = ROAD_SPACING

    # Data collection
    grid_history = []
    metrics_history = {
        'step': [],
        'population': [],
        'pollution': [],
        'population_cap': [],
        'road_count': [],
        'residence_count': [],
        'industry_count': [],
        'service_count': [],
        'greenery_count': [],
        'barren_count': [],
        'phase': []
    }

    # Action trace collection
    action_trace = []

    # Record initial road placement (step 0)
    init_row, init_col = init_road_tile
    init_action_index = init_row * GRID_WIDTH + init_col
    action_trace.append({
        "step": 1,
        "action_index": int(init_action_index),
        "tile_value": int(TileTypes.ROAD.value),
        "tile_type": "ROAD",
        "row": int(init_row),
        "col": int(init_col),
        "total_population": float(agent.total_population),
        "total_pollution": float(agent.total_pollution)
    })

    original_place = agent.place
    def tracked_place(row, col, tile):
        original_place(row, col, tile)
        action_index = row * GRID_WIDTH + col
        action_trace.append({
            "step": int(len(action_trace)) + 1,  # Step number
            "action_index": int(action_index),
            "tile_value": int(tile),
            "tile_type": get_tile_type_name(tile),
            "row": int(row),
            "col": int(col),
            "total_population": float(agent.total_population),
            "total_pollution": float(agent.total_pollution)
        })
    agent.place = tracked_place

    # Run simulation
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION...")
    print("=" * 70)

    for step in range(NUM_STEPS):
        # Store grid state
        grid_history.append(model.grid.tile._mesa_data.copy())

        # Collect metrics
        metrics_history['step'].append(step)
        metrics_history['population'].append(agent.total_population)
        metrics_history['pollution'].append(agent.total_pollution)
        metrics_history['population_cap'].append(agent.population_cap)
        metrics_history['road_count'].append(np.sum(model.road_tiles))
        metrics_history['residence_count'].append(np.sum(model.residence_tiles))
        metrics_history['industry_count'].append(np.sum(model.industry_tiles))
        metrics_history['service_count'].append(np.sum(model.service_tiles))
        metrics_history['greenery_count'].append(np.sum(model.greenery_tiles))
        metrics_history['barren_count'].append(
            np.sum(model.grid.tile._mesa_data == 0)
        )
        metrics_history['phase'].append(agent.phase)

        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{NUM_STEPS} | Phase: {agent.phase} | "
                  f"Pop: {agent.total_population:.0f} | Poll: {agent.total_pollution:.0f}")

        # Step the model
        model.step()

    # Store final state
    grid_history.append(model.grid.tile._mesa_data.copy())

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return model, agent, grid_history, metrics_history, action_trace


def print_final_statistics(model, agent, metrics_history):
    """Print final statistics and analysis."""

    total_tiles = model.width * model.height

    # Final tile counts
    road_count = np.sum(model.road_tiles)
    residence_count = np.sum(model.residence_tiles)
    industry_count = np.sum(model.industry_tiles)
    service_count = np.sum(model.service_tiles)
    greenery_count = np.sum(model.greenery_tiles)
    barren_count = np.sum(model.grid.tile._mesa_data == 0)

    print("\n--- Final City Statistics ---")
    print(f"Total tiles: {total_tiles}")
    print(f"\nTile Distribution:")
    print(f"  Roads:      {road_count:4d} ({road_count/total_tiles:6.1%}) [Target: {TARGET_ROAD_RATIO:.1%}]")
    print(f"  Residences: {residence_count:4d} ({residence_count/total_tiles:6.1%}) [Target: {TARGET_RESIDENCE_RATIO:.1%}]")
    print(f"  Industry:   {industry_count:4d} ({industry_count/total_tiles:6.1%}) [Target: {TARGET_INDUSTRY_RATIO:.1%}]")
    print(f"  Services:   {service_count:4d} ({service_count/total_tiles:6.1%}) [Target: {TARGET_SERVICE_RATIO:.1%}]")
    print(f"  Greenery:   {greenery_count:4d} ({greenery_count/total_tiles:6.1%}) [Target: {TARGET_GREENERY_RATIO:.1%}]")
    print(f"  Barren:     {barren_count:4d} ({barren_count/total_tiles:6.1%})")

    print(f"\nPerformance Metrics:")
    print(f"  Total population:   {agent.total_population:,.0f}")
    print(f"  Total pollution:    {agent.total_pollution:,.0f}")
    print(f"  Population cap:     {agent.population_cap:,.0f}")
    print(f"  Pop/Poll ratio:     {agent.total_population / max(agent.total_pollution, 1):.2f}")

    print(f"\nConnected Clusters:")
    print(f"  Clusters adjacent to roads: {len(model.cluster_adjacent_to_road)}")

    # Calculate ratio deviations
    deviations = {
        'Roads': abs(road_count/total_tiles - TARGET_ROAD_RATIO),
        'Residences': abs(residence_count/total_tiles - TARGET_RESIDENCE_RATIO),
        'Industry': abs(industry_count/total_tiles - TARGET_INDUSTRY_RATIO),
        'Services': abs(service_count/total_tiles - TARGET_SERVICE_RATIO),
        'Greenery': abs(greenery_count/total_tiles - TARGET_GREENERY_RATIO),
    }

    print(f"\nTarget Achievement (Lower deviation = better):")
    for tile_type, deviation in deviations.items():
        print(f"  {tile_type:12s}: {deviation:6.1%} deviation from target")

    avg_deviation = sum(deviations.values()) / len(deviations)
    print(f"  Average deviation: {avg_deviation:6.1%}")


def visualize_results(grid_history, metrics_history, model):
    """Create visualizations of the simulation results."""

    print("\n--- Generating Visualizations ---")

    # Get collected data from model
    collected_model_data = model.data_collectors.get_model_vars_dataframe()
    collected_agent_data = model.data_collectors.get_agent_vars_dataframe()

    # 1. Plot metrics using existing visualization function
    from visualisation import plot_selected_columns

    print("Plotting model metrics...")
    plot_selected_columns(collected_model_data,
                         ["Population Gain", "Pollution Gain", "Total Residence",
                          "Total Industries", "Total Greenery", "Total Service", "Total Road"])
    plt.tight_layout()
    plt.savefig('balanced_city_model_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Model metrics saved to: balanced_city_model_metrics.png")
    plt.close()

    print("Plotting agent metrics...")
    plot_selected_columns(collected_agent_data,
                         ["Total Population", "Total Pollution", "Population Cap"])
    plt.tight_layout()
    plt.savefig('balanced_city_agent_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Agent metrics saved to: balanced_city_agent_metrics.png")
    plt.close()

    # 2. Create GIF animation using existing visualization function
    print("Generating GIF animation...")
    grid_array = np.array(grid_history)
    render_3d_array_to_gif(
        grid_array,
        output_path=OUTPUT_GIF,
        pixel_size=PIXEL_SIZE,
        duration=GIF_DURATION
    )
    print(f"GIF animation saved to: {OUTPUT_GIF}")

    # 3. Additional metrics plot
    df = pd.DataFrame(metrics_history)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Balanced City Planner - Performance Metrics', fontsize=16, fontweight='bold')

    # Plot 1: Population and Pollution over time
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(df['step'], df['population'], 'b-', linewidth=2, label='Population')
    ax1_twin.plot(df['step'], df['pollution'], 'r-', linewidth=2, label='Pollution')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Population', color='b')
    ax1_twin.set_ylabel('Pollution', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Population vs Pollution Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tile counts over time
    ax2 = axes[0, 1]
    ax2.plot(df['step'], df['road_count'], label='Roads', linewidth=2)
    ax2.plot(df['step'], df['residence_count'], label='Residences', linewidth=2)
    ax2.plot(df['step'], df['industry_count'], label='Industry', linewidth=2)
    ax2.plot(df['step'], df['service_count'], label='Services', linewidth=2)
    ax2.plot(df['step'], df['greenery_count'], label='Greenery', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Tile Count')
    ax2.set_title('Tile Counts Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Population capacity utilization
    ax3 = axes[1, 0]
    utilization = (df['population'] / df['population_cap'].replace(0, 1)) * 100
    ax3.plot(df['step'], utilization, 'g-', linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Utilization (%)')
    ax3.set_title('Population Capacity Utilization')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])

    # Plot 4: Barren tiles (remaining space)
    ax4 = axes[1, 1]
    ax4.plot(df['step'], df['barren_count'], 'm-', linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Barren Tiles')
    ax4.set_title('Remaining Empty Space')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('balanced_city_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Metrics plot saved to: balanced_city_metrics.png")
    plt.close()

    # 3. Final city visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    from visualisation import cmap

    im = ax.imshow(grid_history[-1], cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
    ax.set_title(f'Final City Layout (Step {len(grid_history)-1})', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add colorbar
    from visualisation import color_mapping_labels
    labels = [label for _, label, _ in color_mapping_labels]
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(labels)),
                        boundaries=np.arange(len(labels) + 1) - 0.5)
    cbar.ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig('balanced_city_final.png', dpi=150, bbox_inches='tight')
    print(f"Final city layout saved to: balanced_city_final.png")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the simulation
    model, agent, grid_history, metrics_history, action_trace = run_simulation()

    # Print final statistics
    print_final_statistics(model, agent, metrics_history)

    # Generate visualizations
    visualize_results(grid_history, metrics_history, model)

    # Save action trace to JSON
    print("\n--- Saving Action Trace ---")
    with open(OUTPUT_TRACE_JSON, 'w') as f:
        json.dump(action_trace, f, indent=2)
    print(f"Action trace saved to: {OUTPUT_TRACE_JSON}")
    print(f"Total actions recorded: {len(action_trace)}")

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print(f"\nOutput files created:")
    print(f"  - {OUTPUT_GIF} (animated GIF)")
    print(f"  - balanced_city_model_metrics.png (model performance charts)")
    print(f"  - balanced_city_agent_metrics.png (agent performance charts)")
    print(f"  - balanced_city_metrics.png (detailed metrics)")
    print(f"  - balanced_city_final.png (final city layout)")
    print(f"  - {OUTPUT_TRACE_JSON} (action trace JSON)")
    print("\nTo adjust the agent's behavior, modify the hyperparameters")
    print("at the top of this script and run again.")
    print("=" * 70)
