import mesa
import numpy as np
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
        road_network_size = 0
        road_sets = np.array((self.width, self.height))
        road_sets.fill(-1)
        other_tile_set = np.array((self.width, self.height))
        other_tile_set.fill(-1)

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
        self.grid.tile._mesa_data[(row_x,col_y)] = tile

        #update road network
        self.update_road_network(row_x, col_y, tile)

    def update_road_network(self, row_x, col_y, tile):
        #Given a new co-ordinate.
        #check to see if it's connected to an existing graph component
        #else, create a new component

        #if road tile, see if it connects to a network
        #else see if this tile connects to any road set
        pass