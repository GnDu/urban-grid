import mesa
import numpy as np
import networkx
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import PropertyLayer
from scipy.ndimage import distance_transform_cdt

import utils

class DefaultUpdateRules:

    def __init__(self):
        self.residence_population_increase = 10
        self.residence_poll_g = 5
        self.residence_pop_g = 1

        self.greenery_pop_g = 2

        self.industry_pop_g = 8

        self.service_pop_g = 4
        self.service_pop_modifier = 0.2
        self.service_coverage = 2

        self.road_pop_g = 0.5

        

    @staticmethod
    def get_barren(data):
        return data == utils.BARREN

    @staticmethod
    def get_residential(data):
        return data == utils.RESIDENCE
    
    @staticmethod
    def get_greenery(data):
        return data == utils.GREENERY

    @staticmethod
    def get_industry(data):
        return data == utils.INDUSTRY

    @staticmethod
    def get_service(data):
        return data == utils.SERVICE

    @staticmethod
    def get_road(data):
        return data == utils.ROAD

    @staticmethod
    def is_x_within_y_coverage(x:np.array, y:np.array, distance:int):
        """
        Returns a binary 2D array where x's elements 
        are within y's element by a distance. Using chevbev's distance
        which includes diagonal.

        Args:
            x: a 2D binary array
            y: a 2D binary array
            distance: grid distance

        Returns:
            A 2D binary array where 1 indicate x's element within y's coverage
        """

        dist = distance_transform_cdt(~y.astype(bool), metric='chessboard')
        affected = x * dist
        return ((affected<=distance) & (affected > 0)).astype(int)

    def apply_rules(self, model):
        
        residence_tiles = model.grid.tile.select_cells(self.get_residential, return_list=False).astype(int)
        greenery_tiles = model.grid.tile.select_cells(self.get_greenery, return_list=False).astype(int)
        industry_tiles = model.grid.tile.select_cells(self.get_industry, return_list=False).astype(int)
        service_tiles = model.grid.tile.select_cells(self.get_service, return_list=False).astype(int)
        road_tiles = model.grid.tile.select_cells(self.get_road, return_list=False).astype(int)

        #calculate population cap
        # residence_tiles * how much each increase
        model.population_cap = self.residence_population_increase * np.count_nonzero(residence_tiles)

        #road connectivity for industry / service activation
        # skip for now
        filtered_industry_tiles = industry_tiles
        filtered_service_tiles = service_tiles

        pop_g_grid:PropertyLayer = model.grid.pop_g
        pop_g_modifiers = np.ones((pop_g_grid.dimensions))

        poll_g_grid:PropertyLayer = model.grid.poll_g


        #we reset every step. Make it easier to update for now. If it's getting slow, we can try
        #to update it with delta
        pop_g_grid.modify_cells(np.zeros_like)
        poll_g_grid.modify_cells(np.zeros_like)

        #calculate base_pop
        #for residence
        pop_g_grid.modify_cells(np.add, self.residence_pop_g*residence_tiles)

        #for greenery
        pop_g_grid.modify_cells(np.add, self.greenery_pop_g*greenery_tiles)

        #for industry
        pop_g_grid.modify_cells(np.add, self.industry_pop_g*filtered_industry_tiles)

        #for service
        pop_g_grid.modify_cells(np.add, self.service_pop_g*filtered_service_tiles)
        #if within 2 cells of residence, times modifier
        service_coverage = self.is_x_within_y_coverage(residence_tiles, filtered_service_tiles, self.service_coverage)
        # print(service_coverage)
        pop_g_modifiers+=service_coverage*self.service_pop_modifier
        
        #for_road
        pop_g_grid.modify_cells(np.add, self.road_pop_g*road_tiles)

        #calculate base_poll

        #total_population basically sum of all pop_g
        if model.population_cap<=0:
            population_modifier = 1.0
        else:
            population_modifier = ((model.population_cap - model.total_population) / model.population_cap)
        
        pop_g_grid.modify_cells(np.multiply, pop_g_modifiers)
        model.total_population = pop_g_grid.aggregate(np.sum) * population_modifier

class CityModel(mesa.Model):

    def __init__(self, agent_class, width, height, update_rules, seed=None):
        super().__init__(seed=seed)
        self.time_step = 0
        self.width = width
        self.height = height
        self.agent_class = agent_class
        self.update_rules = update_rules

        self.agent_class.create_agents(model=self, n=1)
        self.total_population = 0
        self.total_pollution = 0
        self.population_cap = 0

        self.grid = OrthogonalMooreGrid(dimensions=(self.width, self.height), random=self.random)

        tile_property_layer = PropertyLayer(
            name = "tile", 
            dimensions=(self.width, self.height), 
            default_value=np.int8(0), 
            dtype=np.int8
        )

        pop_g_layer = PropertyLayer(
            name= "pop_g", 
            dimensions=(self.width, self.height), 
            default_value=np.float64(0.0),
            dtype=np.float64
        )
        poll_g_layer = PropertyLayer(
            name="poll_g", 
            dimensions=(self.width, self.height), 
            default_value=np.float64(0.0), 
            dtype=np.float64
        )

        self.grid.add_property_layer(tile_property_layer)
        self.grid.add_property_layer(pop_g_layer)
        self.grid.add_property_layer(poll_g_layer)

    def step(self):
        self.time_step+=1
        self.agents.do("decide")
        #update the environment based on agent decision
        self.update_rules.apply_rules(self)
        #update any internal states, like utiity, etc
        self.agents[0].update()

    def print_stats(self):
        print(f"{'='*5}Stats{'='*5}")
        print(f"Population: {self.total_population} / {self.population_cap}")
        print(f"Pollution: {self.total_pollution}")

    def set_tile(self, x, y, tile):
        #because mesa has a really convulted way to set discrete_space property directly
        #so we are just going to access its private attribute directly.
        self.grid.tile._mesa_data[(x,y)] = tile

        #update road network

    def update_pop_and_poll(self):
        #run through the grid and update total population and pollution

        #first update population cap

        #next check to see if industry and services are 
        pass