import mesa
import numpy as np
from collections import defaultdict
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import PropertyLayer

from utils import TileTypes

#for data collection purpose
def get_city_layout(model):
    return model.grid.tile._mesa_data

def get_population_gain(model):
    return model.update_rules.curr_pop_g

def get_pollution_gain(model):
    return model.update_rules.curr_poll_g

def get_residence_tiles_count(model):
    return np.count_nonzero(model.residence_tiles)

def get_greenery_tiles_count(model):
    return np.count_nonzero(model.greenery_tiles)

def get_industry_tiles_count(model):
    return np.count_nonzero(model.industry_tiles)

def get_service_tiles_count(model):
    return np.count_nonzero(model.service_tiles)

def get_road_tiles_count(model):
    return np.count_nonzero(model.road_tiles)

def get_population_gain_grid(model):
    return model.grid.pop_g._mesa_data

def get_pollution_gain_grid(model):
    return model.grid.poll_g._mesa_data

class CityModel(mesa.Model):

    def __init__(self, agent_class, width, height, update_rules, collect_rate = 1.0, seed=None):
        super().__init__(seed=seed)
        self.time_step = 0
        self.width = width
        self.height = height
        self.agent_class = agent_class
        self.update_rules = update_rules

        self.curr_pop_g = 0
        self.curr_poll_g = 0
        self.population_cap = 0

        self.agent_class.create_agents(model=self, n=1)

        # for road network
        self.road_set_id = 1
        self.road_sets = np.zeros((self.width, self.height)) 
        self.road_adj_to_residence = defaultdict(set)
        self.road_adj_to_industries = defaultdict(set)
        self.road_adj_to_services = defaultdict(set)

        self.grid = OrthogonalMooreGrid(dimensions=(self.width, self.height), random=self.random)

        # defining the various properties

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

        # for book keeping

        self.residence_tiles = np.zeros(self.grid.tile.dimensions)
        self.greenery_tiles = np.zeros(self.grid.tile.dimensions)
        self.industry_tiles = np.zeros(self.grid.tile.dimensions)
        self.service_tiles = np.zeros(self.grid.tile.dimensions)
        self.road_tiles = np.zeros(self.grid.tile.dimensions)

        #defining data collection
        self.collect_rate = collect_rate
        self.data_collectors = mesa.DataCollector(
            model_reporters={"City": get_city_layout,
                             "Population Gain": get_population_gain,
                             "Pollution Gain": get_pollution_gain, 
                             "Population Gain Grid": get_population_gain_grid,
                             "Pollution Gain Grid": get_pollution_gain_grid,
                             "Total Residence": get_residence_tiles_count,
                             "Total Industries": get_industry_tiles_count,
                             "Total Greenery": get_greenery_tiles_count,
                             "Total Service": get_service_tiles_count,
                             "Total Road": get_road_tiles_count},
            agent_reporters={"Total Population": "total_population",
                             "Total Pollution": "total_pollution",
                             "Population Cap": "population_cap"}
        )

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

    def book_keep(self):
        #just a bunch of methods so that agents/update-rules can use them
        self.residence_tiles = self.grid.tile.select_cells(lambda data: data == TileTypes.RESIDENCE.value, 
                                                            return_list=False).astype(int)
        self.greenery_tiles = self.grid.tile.select_cells(lambda data: data == TileTypes.GREENERY.value, 
                                                            return_list=False).astype(int)
        self.industry_tiles = self.grid.tile.select_cells(lambda data: data == TileTypes.INDUSTRY.value, 
                                                            return_list=False).astype(int)
        self.service_tiles = self.grid.tile.select_cells(lambda data: data == TileTypes.SERVICE.value, 
                                                            return_list=False).astype(int)
        self.road_tiles = self.grid.tile.select_cells(lambda data: data == TileTypes.ROAD.value, 
                                                            return_list=False).astype(int)

    def get_city_planner(self):
        return self.agents[0]

    def get_tile(self, row_x, col_y):
        return self.grid.tile._mesa_data[(row_x,col_y)]

    def set_tile(self, row_x, col_y, tile):
        #because mesa has a really convulted way to set discrete_space property directly
        #so we are just going to access its private attribute directly.
        self.grid.tile._mesa_data[row_x, col_y] = tile

        if tile==TileTypes.ROAD.value:
            self.update_road_network(row_x, col_y)
        elif tile==TileTypes.RESIDENCE.value:
            #check if residence is connected to any road network
            self.update_residence_adjacencies(row_x, col_y)
        elif tile==TileTypes.INDUSTRY.value:
            self.update_industry_adjacencies(row_x, col_y)
        elif tile==TileTypes.SERVICE.value:
            self.update_service_adjacencies(row_x, col_y)

    def update_industry_adjacencies(self, row_x, col_y):
        self.update_adjacencies_to_road(row_x, col_y, TileTypes.INDUSTRY.value, self.road_adj_to_industries)

    def update_service_adjacencies(self, row_x, col_y):
        self.update_adjacencies_to_road(row_x, col_y, TileTypes.SERVICE.value, self.road_adj_to_services)

    def update_residence_adjacencies(self, row_x, col_y):
        self.update_adjacencies_to_road(row_x, col_y, TileTypes.RESIDENCE.value, self.road_adj_to_residence)

    def update_adjacencies_to_road(self, row_x, col_y, tile_value:int, road_adj_dict:defaultdict):
        """
        Given a residence co-ordinate (row_x, col_y), 
        check if there are any road network in its neighbourhood.

        If there are, update the road network adjacencies

        """

        #sanity check
        assert(int(self.grid.tile._mesa_data[(row_x, col_y)])==tile_value)

        neighbours, _ = self.get_neighbours(self.road_sets, row_x, col_y, is_4_neighbourhood=True)
        neighbours = set(neighbours.flatten().tolist())
        try:
            neighbours.remove(0)
        except KeyError:
            pass

        for neighbour in neighbours:
            self.road_adj_dict[neighbour].add((row_x, col_y))

    def update_road_adjacencies(self, row_x, col_y, tiletype_value, curr_road_id):
        tiles = self.grid.tile.select_cells(lambda data: data == tiletype_value, 
                                            return_list=False).astype(int)
        neighbours, center = self.get_neighbours(tiles, row_x, col_y, is_4_neighbourhood=True)
        local_coordinates = np.argwhere(neighbours>0)
        if local_coordinates.size > 0:
            #using center, find the relative offset and use the x_row, y_col to find the actual
            #coordinates
            residence_coordinates = np.array([row_x, col_y]) - (center - local_coordinates)
            for residence_coordinate in residence_coordinates:
                self.road_adj_to_residence[curr_road_id].add((int(residence_coordinate[0]), 
                                                                int(residence_coordinate[1])))

    def update_road_network(self, row_x, col_y):
        # simplyfying this
        # for every road tile added we check if they can be 
        # part of a bigger network
        # we also check if any residence is adjacent to placed tile
        
        #sanity check
        assert(int(self.grid.tile._mesa_data[(row_x, col_y)])==TileTypes.ROAD.value)

        neighbours, _ = self.get_neighbours(self.road_sets, row_x, col_y, is_4_neighbourhood=True)
        neighbours = set(neighbours.flatten().tolist())
        
        #raod_network_id must be >0
        #remove any invalid ids
        try:
            neighbours.remove(0)
        except KeyError:
            pass

        curr_road_id = None
        if len(neighbours)==0:
            #if there are no neighbours, create a new road network
            
            self.road_sets[row_x, col_y] = self.road_set_id
            
            curr_road_id = self.road_set_id
            self.road_set_id+=1
        else:
            #if there are more than one network, merge them.
            #the smaller network should be subsumed under the bigger one

            #determine who has the bigger network
            max_count = -1
            max_neighbour = -1
            for neighbour in neighbours:
                total_in_network = (self.road_sets==neighbour).sum()
                if total_in_network > max_count:
                    max_count = total_in_network
                    max_neighbour = neighbour

            #set the new tile to the one with the biggest network
            self.road_sets[row_x, col_y] = max_neighbour
            curr_road_id = max_neighbour
            #now set the rest of the neighour network to that max_neighbour network
            for neighbour in neighbours:
                if neighbour==max_neighbour:
                    continue
                self.road_sets[self.road_sets==neighbour] = max_neighbour
                #for network adjacent to residence, max_neighbour is also adjacent to them
                self.road_adj_to_residence[max_neighbour]|=self.road_adj_to_residence[neighbour]
                #we remove all residences of the old neighbour
                del self.road_adj_to_residence[neighbour]
            
            #now check and see if a road is adjacent to a residence
            self.update_road_adjacencies(row_x, col_y, TileTypes.RESIDENCE.value, curr_road_id)
            self.update_road_adjacencies(row_x, col_y, TileTypes.INDUSTRY.value, curr_road_id)
            self.update_road_adjacencies(row_x, col_y, TileTypes.SERVICE.value, curr_road_id)
       

    @staticmethod
    def get_neighbours(tiles:np.array, center_row:int, center_col:int, is_4_neighbourhood:bool=False):
        """
            Get the direct neighbour of (center_row, center_col)
            by default, this will return the moore neighbours (8 neighbours): 
            https://en.wikipedia.org/wiki/Moore_neighborhood
            
            If `is_4_neighbour` is true, return the Von Neumann neighbourhood.
            https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
            
            Returns the neighbourhood of the cell (could be 3x3, 3x2, 2x3 or 2x2) and 
            the co-ordinate of the center   
        """


        width, height = tiles.shape
        #check to see if it connects to any road set
        x_lower = center_row-1
        x_upper = center_row+2
        y_lower = center_col-1
        y_upper = center_col+2
        center = [1,1]

        #check for edge case, literally.
        #if x_lower is negative, that means it is at the 
        # top edge
        if x_lower<0:
            x_lower = 0
            center[0] = 0
        
        #if x_upper is more than width, that means it is at
        #bottom edge
        if x_upper>=width:
            x_upper = width
            center[0] = 1

        #if y_lower is negative, that means it is at
        #the left edge
        if y_lower<0:
            y_lower = 0
            center[1] = 0

        #if y_lower is negative, that means it is at
        #the right edge
        if y_upper >= height:
            y_upper = height
            center[1] = 1

        neighbours = np.copy(tiles[x_lower:x_upper, y_lower:y_upper])
        neighbours[tuple(center)] = 0

        #we get diagonals
        if is_4_neighbourhood:
            diagonal_adjacents = [[],[]]
            for r in [-1,1]:
                for c in [-1,1]:
                    r_i = center[0] + r
                    c_i = center[1] + c
                    if r_i>=0 and r_i < neighbours.shape[0] \
                        and c_i>=0 and c_i < neighbours.shape[1]:
                        diagonal_adjacents[0].append(r_i)
                        diagonal_adjacents[1].append(c_i)

            neighbours[diagonal_adjacents[0], diagonal_adjacents[1]] = 0
        
        return neighbours, np.array(center)