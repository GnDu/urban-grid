from typing import List, Dict, Tuple, Any
import numpy as np

GREENERY=0
ROAD=1
INDUSTRY=2
RESIDENCE=3

class SimpleRuleset:
    """
    This simple ruleset simply does the following:
    - calculate population cap based on the number of residence
    """
    def __init__(self):
        pass

    def calculate_metric(self, environment:"CityEnvironment"):
        pass

class SimpleTermination():
    """
    This simple ruleset simply does the following:
    - calculate population cap based on the number of residence
    """
    def __init__(self):
        pass

    def is_terminate(self, environment:"CityEnvironment"):
        pass

class CityEnvironment:

    def __init__(self, width, height, rule_set=None):
        self.grid = np.zeros((height, width))

        #if rule-set is not given, use the default
        self.rule_set = SimpleRuleset() if rule_set is None else rule_set
        
        #metric
        self.population=0
        self.population_cap=0
        self.pollution=0

        #time step
        self.time = 0

    def forward(self, decisions:Dict[Tuple, int]):
        #make the decision
        for grid_coords, cell_type in decisions.items():
            print(grid_coords, cell_type)
            self.grid[grid_coords] = cell_type

        #forward the time
        self.time+=1

        #calculate metric, the rule_set can use everything within the environment.
        self.population, self.pollution = self.rule_set.calculate_metric(self)