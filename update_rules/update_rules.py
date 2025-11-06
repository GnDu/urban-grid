# update_rules/update_rules.py
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import List
from mesa.discrete_space.property_layer import PropertyLayer
from scipy.ndimage import distance_transform_cdt, grey_dilation, generate_binary_structure, iterate_structure

ZERO_EPS = 1e-8

@dataclass
class DefaultUpdateRulesParameters:
    residence_population_increase: int
    residence_poll_g: float
    residence_pop_g: float
    residence_walking_distance: int

    greenery_poll_minus: float
    greenery_pop_g: float
    greenery_coverage: int

    industry_poll_g: float
    industry_pop_g: float
    industry_coverage: int
    industry_connectivity_initial_modifier: float
    industry_connectivity_cap: int
    
    service_poll_g: float
    service_pop_g: float
    service_pop_modifier: float
    service_coverage: int
    service_connectivity_initial_modifier: float
    service_connectivity_cap: int

    road_poll_g: float
    road_pop_g: float


class DefaultUpdateRules:
    def __init__(self):
        self.curr_pop_g = 0
        self.curr_poll_g = 0
        self.population_cap = 0

    def clone(self):
        cloned_obj = DefaultUpdateRules()
        cloned_obj.curr_pop_g = self.curr_pop_g
        cloned_obj.curr_poll_g = self.curr_poll_g
        cloned_obj.population_cap = self.population_cap
        cloned_obj.set_parameters(self.parameter_saved)
        return cloned_obj

    def set_parameters(self, parameters: DefaultUpdateRulesParameters):
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
        self.industry_connectivity_initial_modifier= parameters.industry_connectivity_initial_modifier
        assert(parameters.industry_connectivity_cap>0)
        self.industry_connectivity_cap = parameters.industry_connectivity_cap
        self.industry_connectivity_modifier_to_cap = (1 - self.industry_connectivity_initial_modifier) / self.industry_connectivity_cap

        self.service_poll_g = parameters.service_poll_g
        self.service_pop_g = parameters.service_pop_g
        self.service_pop_modifier = parameters.service_pop_modifier
        self.service_coverage = parameters.service_coverage
        self.service_connectivity_initial_modifier = parameters.service_connectivity_initial_modifier
        assert(parameters.service_connectivity_cap > 0)
        self.service_connectivity_cap = parameters.service_connectivity_cap
        self.service_connectivity_modifier_to_cap = (1 - self.service_connectivity_initial_modifier) / self.service_connectivity_cap

        self.road_poll_g  = parameters.road_poll_g
        self.road_pop_g  = parameters.road_pop_g

    @staticmethod
    def is_x_within_y_coverage(x:np.array, y:np.array, distance:int, metric='chessboard'):
        dist = distance_transform_cdt(~y.astype(bool), metric)
        affected = x * dist
        return ((affected<=distance) & (affected > 0)).astype(int)

    @staticmethod
    def flat_dilation(grid:np.array, coverage:int, base_value:float=None):
        st = generate_binary_structure(2,1)
        coverage = grey_dilation(grid, footprint = iterate_structure(st,coverage), mode='constant')
        if base_value is not None:
            coverage[coverage>0]=base_value
        return coverage

    def find_linked_residences(self, model, road_tile_adjacent_list, resident_tiles, tiles_of_interest):
        road_ids = model.road_adj_to_residence.keys()
        tile_of_interest_road_ids = road_tile_adjacent_list.keys()
        connected_ids = road_ids & tile_of_interest_road_ids

        tile_connected_to_residence = defaultdict(set)
        unlinked_residence_tiles = np.copy(resident_tiles)
        linked_residence_coordinates = [[],[]]
        for road_id in connected_ids:
            tiles_conncected = road_tile_adjacent_list[road_id]
            for tile_coordinate in tiles_conncected:
                tile_connected_to_residence[tile_coordinate]|=model.road_adj_to_residence[road_id]
                for r, c in model.road_adj_to_residence[road_id]:
                    linked_residence_coordinates[0].append(r)
                    linked_residence_coordinates[1].append(c)

        unlinked_residence_tiles[linked_residence_coordinates] = 0
        tile_coordinates = np.argwhere(tiles_of_interest>0)
        st = generate_binary_structure(2,1)
        for tile_coordinate in tile_coordinates:
            tile_coordinate = (int(tile_coordinate[0]), int(tile_coordinate[1]))
            mask = np.zeros(tiles_of_interest.shape)
            mask[tile_coordinate]=1
            mask = grey_dilation(mask, footprint = iterate_structure(st,self.residence_walking_distance), mode='constant')
            linked_residences = unlinked_residence_tiles * mask
            residence_coordinates = np.argwhere(linked_residences>0)
            residence_coordinates = set([(int(r), int(c)) for r,c in residence_coordinates])
            tile_connected_to_residence[tile_coordinate]|=residence_coordinates

        updated_row = []
        updated_col = []
        residences_count = []
        for tile_coordinate, residence_coordinates in tile_connected_to_residence.items():
            residence_count = len(residence_coordinates)
            updated_row.append(tile_coordinate[0])
            updated_col.append(tile_coordinate[1])
            residences_count.append(residence_count)

        return updated_row, updated_col, residences_count
        
    def calculate_connectivity_modifier(self, linked_residence_count, initial_modifier, connectivity_to_cap_modifier):
        linked_tile_poll_modifier = []
        linked_tile_pop_modifier = []
        for count in linked_residence_count:
            poll_modifier = initial_modifier + count*connectivity_to_cap_modifier
            pop_modifier = initial_modifier + count*connectivity_to_cap_modifier
            linked_tile_poll_modifier.append(poll_modifier)
            linked_tile_pop_modifier.append(pop_modifier)
        return linked_tile_pop_modifier, linked_tile_poll_modifier

    def apply_rules(self, model):
        residence_tiles = model.residence_tiles
        greenery_tiles = model.greenery_tiles
        industry_tiles = model.industry_tiles
        service_tiles = model.service_tiles
        road_tiles = model.road_tiles

        population_cap = self.residence_population_increase * np.count_nonzero(residence_tiles)

        linked_industries_row, linked_industries_col, linked_industries_count = self.find_linked_residences(
            model, model.road_adj_to_industries, residence_tiles, industry_tiles)
        linked_industries_pop_modifier, linked_industries_poll_modifier = self.calculate_connectivity_modifier(
            linked_industries_count, self.industry_connectivity_initial_modifier, self.industry_connectivity_modifier_to_cap)

        linked_services_row, linked_services_col, linked_services_count = self.find_linked_residences(
            model, model.road_adj_to_services, residence_tiles, service_tiles)
        linked_services_pop_modifier, linked_services_poll_modifier = self.calculate_connectivity_modifier(
            linked_services_count, self.service_connectivity_initial_modifier, self.service_connectivity_modifier_to_cap)

        linked_service_tiles = np.zeros(service_tiles.shape)
        linked_service_tiles[linked_services_row, linked_services_col] = 1

        industry_tiles_pop_modifier = (industry_tiles * self.industry_connectivity_initial_modifier).astype(np.float64)
        industry_tiles_pop_modifier[linked_industries_row, linked_industries_col] = linked_industries_pop_modifier
        industry_tiles_poll_modifier = (industry_tiles * self.industry_connectivity_initial_modifier).astype(np.float64)
        industry_tiles_poll_modifier[linked_industries_row, linked_industries_col] = linked_industries_poll_modifier
        industry_poll_g = self.industry_poll_g*industry_tiles_pop_modifier

        service_tiles_pop_modifier = (service_tiles * self.service_connectivity_initial_modifier).astype(np.float64)
        service_tiles_pop_modifier[linked_services_row, linked_services_col] = linked_services_pop_modifier
        service_tiles_poll_modifier = (service_tiles * self.service_connectivity_initial_modifier).astype(np.float64)
        service_tiles_poll_modifier[linked_services_row, linked_services_col] = linked_services_poll_modifier

        pop_g_grid:PropertyLayer = model.grid.pop_g
        poll_g_grid:PropertyLayer = model.grid.poll_g

        pop_g_grid.modify_cells(np.zeros_like)
        poll_g_grid.modify_cells(np.zeros_like)

        service_pop_modifier = np.ones(service_tiles.shape, dtype=np.float64)
        service_coordinates = np.argwhere(service_tiles>0)
        for service_coordinate in service_coordinates:
            st = generate_binary_structure(2,1)
            coverage = np.zeros(service_tiles.shape, dtype=np.float64)
            coverage[service_coordinate[0],service_coordinate[1]]=1
            coverage = grey_dilation(coverage, footprint = iterate_structure(st,self.service_coverage), mode='constant')
            coverage[coverage>0]=self.service_pop_modifier
            service_pop_modifier+=coverage

        pop_g_grid.modify_cells(np.add, self.residence_pop_g*residence_tiles*service_pop_modifier)
        pop_g_grid.modify_cells(np.add, self.greenery_pop_g*greenery_tiles)
        pop_g_grid.modify_cells(np.add, self.industry_pop_g*industry_tiles_pop_modifier)
        pop_g_grid.modify_cells(np.add, self.service_pop_g*service_tiles_pop_modifier)
        pop_g_grid.modify_cells(np.add, self.road_pop_g*road_tiles)

        poll_g_grid.modify_cells(np.add, self.residence_poll_g*residence_tiles)
        industry_poll_g = self.flat_dilation(industry_poll_g, self.industry_coverage)
        poll_g_grid.modify_cells(np.add, industry_poll_g)
        poll_g_grid.modify_cells(np.add, self.service_poll_g*service_tiles_poll_modifier)
        poll_g_grid.modify_cells(np.add, self.road_poll_g*road_tiles)

        greenery_poll_modifier = np.ones(greenery_tiles.shape)
        greenery_coordinates = np.argwhere(greenery_tiles>0)
        for greenery_coordinate in greenery_coordinates:
            st = generate_binary_structure(2,1)
            coverage = np.zeros(greenery_tiles.shape, dtype=np.float64)
            coverage[greenery_coordinate[0], greenery_coordinate[1]]=1
            coverage = grey_dilation(coverage, footprint = iterate_structure(st,self.greenery_coverage), mode='constant')
            coverage[coverage>0]=self.greenery_poll_minus
            greenery_poll_modifier+=coverage
        greenery_poll_modifier = 1 / greenery_poll_modifier
        poll_g_grid.modify_cells(np.multiply, greenery_poll_modifier)
        poll_g_grid.modify_cells(lambda x: 0, condition=lambda grid: grid<=0)

        if population_cap<=0:
            population_modifier = population_cap - model.get_city_planner().total_population
        else:
            diff = population_cap - model.get_city_planner().total_population
            if diff<=ZERO_EPS:
                diff = 0
            population_modifier = diff / population_cap
        
        self.population_cap = float(population_cap)
        self.curr_pop_g = float(pop_g_grid.aggregate(np.sum) * population_modifier)
        self.curr_poll_g = float(poll_g_grid.aggregate(np.sum))
        return float(self.population_cap), float(self.curr_pop_g), float(self.curr_poll_g)
