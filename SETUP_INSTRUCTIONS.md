# Setup Instructions

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `mesa[rec]==3.3` - Agent-based modeling framework
- `jupyterlab==4.4.9` - For running notebooks
- `scikit-learn==1.7` - Machine learning utilities
- `matplotlib>=3.5.0` - Plotting and visualization
- `pandas>=1.3.0` - Data analysis
- `imageio>=2.9.0` - GIF generation
- `pillow>=8.0.0` - Image processing
- `ipywidgets>=7.6.0` - Interactive widgets

## Running the Balanced City Planner

Once dependencies are installed, run the test script:

```bash
python test_balanced_agent.py
```

This will:
1. Load configuration from `data/update_parameters/UpdateRule_Stricter_Road.json`
2. Run a 300-step simulation with the BalancedCityPlanner agent
3. Generate visualizations and statistics

## Configuration Files

The test script uses **UpdateRule_Stricter_Road.json** which has these settings:

- `residence_population_increase: 300` (vs 30 in default)
- `greenery_poll_minus: 0.9` (vs 0.05 in default) - Much stronger pollution reduction
- `industry_connectivity_initial_modifier: 0` - Industries start at 0% efficiency until connected
- `service_connectivity_initial_modifier: 0` - Services start at 0% efficiency until connected
- `residence_walking_distance: 0` - NO walking allowed, must use roads

These stricter rules enforce the constraint that **all tiles must be connected to roads to be functional**.

## Hyperparameter Tuning

Edit the top of `test_balanced_agent.py` to adjust agent behavior:

```python
# Target tile ratios
TARGET_ROAD_RATIO = 0.15
TARGET_RESIDENCE_RATIO = 0.20
TARGET_INDUSTRY_RATIO = 0.20
TARGET_SERVICE_RATIO = 0.20
TARGET_GREENERY_RATIO = 0.25

# Strategy parameters
ROAD_EXPANSION_AGGRESSIVE = True
USE_ZONING = True
ROAD_SPACING = 4
POLLUTION_THRESHOLD = 100
```

## Output Files

After running, you'll get:
- `balanced_city_animation.html` - Interactive city evolution animation
- `balanced_city_metrics.png` - Performance charts
- `balanced_city_final.png` - Final city layout

Open the HTML file in any web browser to see the animated city development.
