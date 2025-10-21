import numpy as np
from dataclasses import dataclass
from mesa.discrete_space.property_layer import PropertyLayer
from scipy.ndimage import distance_transform_cdt, grey_dilation, \
                            generate_binary_structure, iterate_structure

ZERO_EPS = 1e-8

@dataclass
class DefaultUpdateRulesParameters:

    residence_population_increase: int
    residence_poll_g: float
    residence_pop_g: float

    greenery_poll_minus: float
    greenery_pop_g: float
    greenery_coverage: int

    industry_poll_g: float
    industry_pop_g: float
    industry_coverage: int
    
    service_poll_g: float
    service_pop_g: float
    service_pop_modifier: float
    service_coverage: int

    road_poll_g: float
    road_pop_g: float

class DefaultUpdateRules:
    
    def __init__(self):

        self.curr_pop_g = 0
        self.curr_poll_g = 0
        self.population_cap = 0

    def set_parameters(self, parameters:DefaultUpdateRulesParameters):
        self.residence_population_increase = parameters.residence_population_increase
        self.residence_poll_g = parameters.residence_poll_g
        self.residence_pop_g = parameters.residence_pop_g

        self.greenery_poll_minus  = parameters.greenery_poll_minus
        self.greenery_pop_g  = parameters.greenery_pop_g
        self.greenery_coverage = parameters.greenery_coverage

        self.industry_poll_g = parameters.industry_poll_g
        self.industry_pop_g = parameters.industry_pop_g
        self.industry_coverage = parameters.industry_coverage
        
        self.service_poll_g = parameters.service_poll_g
        self.service_pop_g = parameters.service_pop_g
        self.service_pop_modifier = parameters.service_pop_modifier
        self.service_coverage = parameters.service_coverage

        self.road_poll_g  = parameters.road_poll_g
        self.road_pop_g  = parameters.road_pop_g

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

    @staticmethod
    def flat_dilation(grid:np.array, coverage:int, base_value:float=None):
        """
        Given a binary grid, find neighbouring cells, where neighbouring cells 
        are defined by a coverage value. If `base_value` is defined, each of 
        this cells will be replaced wit the base_value instead, else, the cells 
        will simply take on value of said cells.

        Note: if the coverage spills over the edge of the grid, they will be ignored.
        """

        st = generate_binary_structure(2,1)
        coverage = grey_dilation(grid, footprint = iterate_structure(st,coverage), mode='constant')
        if base_value is not None:
            #we set everyhing to base_value regardless of values.
            coverage[coverage>0]=base_value
        return coverage

    def apply_rules(self, model):
        
        residence_tiles = model.residence_tiles
        greenery_tiles = model.greenery_tiles
        industry_tiles = model.industry_tiles
        service_tiles = model.service_tiles
        road_tiles = model.road_tiles

        #calculate population cap
        # residence_tiles * how much each increase
        population_cap = self.residence_population_increase * np.count_nonzero(residence_tiles)

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

        ################### calculate base_pop ################### 
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

        ################### calculate base_poll ################### 
        #for residence
        poll_g_grid.modify_cells(np.add, self.residence_poll_g*residence_tiles)

        #for industry
        #get the industry pollution coverage
        industry_poll_g = self.flat_dilation(filtered_industry_tiles, self.industry_coverage, self.industry_poll_g)
        poll_g_grid.modify_cells(np.add, industry_poll_g)

        #for service
        poll_g_grid.modify_cells(np.add, self.service_poll_g*filtered_service_tiles)

        #for road
        poll_g_grid.modify_cells(np.add, self.road_poll_g*road_tiles)

        #for greenery
        #pollution mitigation
        greenery_coverage= self.flat_dilation(greenery_tiles, self.greenery_coverage, self.greenery_poll_minus)
        poll_g_grid.modify_cells(np.subtract, greenery_coverage)
        #if any poll_g go negative, set them to zero
        poll_g_grid.modify_cells(lambda x: 0, condition=lambda grid: grid<0)

        #total_population basically sum of all pop_g
        if population_cap<=0:
            population_modifier = 1.0
        else:
            diff = population_cap - model.get_city_planner().total_population
            #if diff is really small, just set it to 0
            if diff<=ZERO_EPS:
                diff = 0
            population_modifier = diff / population_cap
        
        pop_g_grid.modify_cells(np.multiply, pop_g_modifiers)
        self.population_cap = population_cap
        self.curr_pop_g = pop_g_grid.aggregate(np.sum) * population_modifier

        #total pollution is basically the sum of all poll_g
        self.curr_poll_g = poll_g_grid.aggregate(np.sum)

        return self.population_cap, self.curr_pop_g, self.curr_poll_g