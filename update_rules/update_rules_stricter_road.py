import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import List
from mesa.discrete_space.property_layer import PropertyLayer
from scipy.ndimage import distance_transform_cdt, grey_dilation, \
                            generate_binary_structure, iterate_structure

from update_rules.update_rules import DefaultUpdateRules, DefaultUpdateRulesParameters
ZERO_EPS = 1e-8

class UpdateRulesStricterRoad(DefaultUpdateRules):
    
    def __init__(self):
        super().__init__()
        self.curr_pop_g = 0
        self.curr_poll_g = 0
        self.population_cap = 0

    def clone(self):
        cloned_obj = UpdateRulesStricterRoad()
        cloned_obj.curr_pop_g = self.curr_pop_g
        cloned_obj.curr_poll_g = self.curr_poll_g
        cloned_obj.population_cap = self.population_cap
        cloned_obj.set_parameters(self.parameter_saved)
        return cloned_obj

    def set_parameters(self, parameters:DefaultUpdateRulesParameters):
        self.parameter_saved = parameters
        self.residence_population_increase = parameters.residence_population_increase
        self.residence_poll_g = parameters.residence_poll_g
        self.residence_pop_g = parameters.residence_pop_g
        self.residence_walking_distance = parameters.residence_walking_distance

        self.greenery_poll_minus  = parameters.greenery_poll_minus
        self.greenery_pop_g  = parameters.greenery_pop_g
        self.greenery_coverage = parameters.greenery_coverage

        self.industry_poll_g = parameters.industry_poll_g
        self.industry_pop_g = parameters.industry_pop_g
        self.industry_coverage = parameters.industry_coverage

        if parameters.industry_connectivity_initial_modifier!=0:
            print("Warning: industry_connectivity_initial_modifier is ignored. Setting to 0")
        self.industry_connectivity_initial_modifier= 0
        
        # connectivity cap refers to how many residences to connect 
        # before the tile acheives 100% population and pollution
        # we will inverse this and calculate how much each residence tile 
        # will contribute 
        assert(parameters.industry_connectivity_cap>0)
        self.industry_connectivity_cap = parameters.industry_connectivity_cap
        self.industry_connectivity_modifier_to_cap = (1 - self.industry_connectivity_initial_modifier) / self.industry_connectivity_cap

        
        self.service_poll_g = parameters.service_poll_g
        self.service_pop_g = parameters.service_pop_g
        self.service_pop_modifier = parameters.service_pop_modifier
        self.service_coverage = parameters.service_coverage

        if parameters.service_connectivity_initial_modifier!=0:
            print("Warning: service_connectivity_initial_modifier is ignored. Setting to 0")
        self.service_connectivity_initial_modifier = 0

        # connectivity cap refers to how many residences to connect 
        # before the tile acheives 100% population and pollution
        # we will inverse this and calculate how much each residence tile 
        # will contribute 
        assert(parameters.service_connectivity_cap > 0)
        self.service_connectivity_cap = parameters.service_connectivity_cap
        self.service_connectivity_modifier_to_cap = (1 - self.service_connectivity_initial_modifier) / self.service_connectivity_cap

        self.road_poll_g  = parameters.road_poll_g
        self.road_pop_g  = parameters.road_pop_g
        
    def calculate_connectivity_modifier(self, connected_residence_tiles_count, connectivity_to_cap_modifier):
        base_modifier = connected_residence_tiles_count * connectivity_to_cap_modifier
        #if a tile is operating at 100% efficiency, the pollution increase slows by a factor of 4
        #Basically if connectivity_to_cap_modifier = 0.2, pollution will only increase by 0.05 after 100% productivity
        if base_modifier>1.0:
            connectivity_cap = 1 / connectivity_to_cap_modifier
            poll_modifier = 1 + (connected_residence_tiles_count - connectivity_cap) * (connectivity_to_cap_modifier / 4)
        else:
            poll_modifier = base_modifier

        return base_modifier, poll_modifier

    def filter_disconnected_tiles(self, tiles, tile_cluster, cluster_adjacent_to_road):
        #we will only return tiles that are in clusters adjacent to road.
        #get all cluster ids connected to road
        valid_cluster_ids = np.array(list(cluster_adjacent_to_road))
        mask = np.isin(tile_cluster, valid_cluster_ids)
        valid_tiles = tiles * mask
        return valid_tiles

    def apply_rules(self, model):

        # residence_tiles = model.residence_tiles
        # greenery_tiles = model.greenery_tiles
        # industry_tiles = model.industry_tiles
        # service_tiles = model.service_tiles
        # road_tiles = model.road_tiles

        residence_tiles = self.filter_disconnected_tiles(model.residence_tiles, model.residence_cluster, model.cluster_adjacent_to_road)
        industry_tiles = self.filter_disconnected_tiles(model.industry_tiles, model.industry_cluster, model.cluster_adjacent_to_road)
        service_tiles = self.filter_disconnected_tiles(model.service_tiles, model.service_cluster, model.cluster_adjacent_to_road)

        road_tiles = model.road_tiles

        connected_greenery_tiles = self.filter_disconnected_tiles(model.greenery_tiles, model.greenery_cluster, model.cluster_adjacent_to_road)

        #calculate population cap
        # residence_tiles * how much each increase
        connected_residence_tiles_count = np.count_nonzero(residence_tiles)
        population_cap = self.residence_population_increase * connected_residence_tiles_count

        # Industry / Service connectivity
        #   - Connectivity are only via roads
        #   - Each cluster will have the same number of residences
        #   - Only residence cluster connected to roads are valid
        #   - Only industry / service cluster connected to roads are valid
        #   - Thus, all valid industry / service cluster will have the same number of residences connected

        industry_prod_pop_modifier, industry_prod_poll_modifier = self.calculate_connectivity_modifier(connected_residence_tiles_count, self.industry_connectivity_modifier_to_cap)
        service_prod_pop_modifier, service_prod_poll_modifier = self.calculate_connectivity_modifier(connected_residence_tiles_count, self.service_connectivity_modifier_to_cap)

        pop_g_grid:PropertyLayer = model.grid.pop_g
        poll_g_grid:PropertyLayer = model.grid.poll_g

        #we reset every step. Make it easier to update for now. If it's getting slow, we can try
        #to update it with delta
        pop_g_grid.modify_cells(np.zeros_like)
        poll_g_grid.modify_cells(np.zeros_like)

        ################### calculate base_pop ################### 
        #for residence
        #if within 2 cells of residence, times modifier
        #get the service that are activated
        service_pop_modifier = np.ones(service_tiles.shape, dtype=np.float64)
        service_coordinates = np.argwhere(service_tiles>0)
        # print("Calculating")
        # print(service_coordinates)
        for service_coordinate in service_coordinates:
            st = generate_binary_structure(2,1)
            coverage = np.zeros(service_tiles.shape, dtype=np.float64)
            coverage[service_coordinate[0],service_coordinate[1]]=1
            coverage = grey_dilation(coverage, footprint = iterate_structure(st,self.service_coverage), mode='constant')
            coverage[coverage>0]=self.service_pop_modifier
            # print(coverage)
            service_pop_modifier+=coverage

        # service_coverage = self.is_x_within_y_coverage(residence_tiles, 
        #                                                 service_tiles, 
        #                                                 self.service_coverage)
        # service_coverage = service_coverage.astype(np.float64) * self.service_pop_modifier
        # service_coverage[service_coverage<=0] = 1
        # print("Result")
        # print(service_pop_modifier)
        pop_g_grid.modify_cells(np.add, self.residence_pop_g*residence_tiles*service_pop_modifier)

        #for greenery
        pop_g_grid.modify_cells(np.add, self.greenery_pop_g*connected_greenery_tiles)

        #for industry
        pop_g_grid.modify_cells(np.add, industry_tiles*self.industry_pop_g*industry_prod_pop_modifier)
        
        #for service
        pop_g_grid.modify_cells(np.add, service_tiles*self.service_pop_g*service_prod_pop_modifier)

        #for_road
        pop_g_grid.modify_cells(np.add, self.road_pop_g*road_tiles)

        ################### calculate base_poll ################### 
        #for residence
        poll_g_grid.modify_cells(np.add, self.residence_poll_g*residence_tiles)

        #for industry
        #get the industry pollution coverage
        industry_poll_g = industry_tiles * industry_prod_poll_modifier
        industry_poll_g = self.flat_dilation(industry_poll_g, 
                                             self.industry_coverage)
        poll_g_grid.modify_cells(np.add, industry_poll_g)

        #for service
        poll_g_grid.modify_cells(np.add, self.service_poll_g * service_tiles * service_prod_poll_modifier)

        #for road
        poll_g_grid.modify_cells(np.add, self.road_poll_g*road_tiles)

        #for greenery
        #pollution mitigation
        greenery_poll_modifier = np.ones(model.greenery_tiles.shape)
        greenery_coordinates = np.argwhere(model.greenery_tiles>0)
        for greenery_coordinate in greenery_coordinates:
            st = generate_binary_structure(2,1)
            coverage = np.zeros(model.greenery_tiles.shape, dtype=np.float64)
            coverage[greenery_coordinate[0], greenery_coordinate[1]]=1
            coverage = grey_dilation(coverage, footprint = iterate_structure(st,self.greenery_coverage), mode='constant')
            coverage[coverage>0]=self.greenery_poll_minus
            greenery_poll_modifier+=coverage

        #this guarantees that it will greenery_poll_modifier < 1 and >0
        greenery_poll_modifier = 1 / greenery_poll_modifier
        # print(greenery_poll_modifier)
        poll_g_grid.modify_cells(np.multiply, greenery_poll_modifier)
        #if any poll_g go negative, set them to zero
        poll_g_grid.modify_cells(lambda x: 0, condition=lambda grid: grid<=0)

        #total_population basically sum of all pop_g
        if population_cap<=0:
            population_modifier = -1
        else:
            diff = population_cap - model.get_city_planner().total_population
            #if diff is really small, just set it to 0
            if diff<=ZERO_EPS:
                diff = 0
            population_modifier = diff / population_cap
        
        self.population_cap = population_cap
        self.curr_pop_g = pop_g_grid.aggregate(np.sum) * population_modifier

        #total pollution is basically the sum of all poll_g
        self.curr_poll_g = poll_g_grid.aggregate(np.sum)

        return float(self.population_cap), float(self.curr_pop_g), float(self.curr_poll_g)