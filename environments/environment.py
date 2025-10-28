import mesa
import networkx as nx
import numpy as np
from collections import defaultdict
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import PropertyLayer
from scipy.ndimage import  grey_dilation, \
                            generate_binary_structure, iterate_structure

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
        self.road_graph = nx.Graph()
        self.road_connected_sets = []

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
        elif tile==TileTypes.BARREN.value:
            #deleting a tile
            #determine which tile is it deleting
            tile_query = self.get_tile(row_x, col_y)
            tile_of_interests = [TileTypes.RESIDENCE.value, TileTypes.INDUSTRY.value, TileTypes.SERVICE.value]
            adj_of_interests = [self.road_adj_to_residence, self.road_adj_to_industries, self.road_adj_to_services]

            if tile_query in tile_of_interests:
                #if road is adjacent to the tile, we need to remove it
                interest_index = tile_of_interests.index(tile_query)
                adj_list = adj_of_interests[interest_index]
                self.remove_tile_from_adjacencies(row_x, col_y, adj_list)
            elif tile_query==TileTypes.ROAD.value:
                #destory road tile
                #update road network, carefully.
                self.remove_road_tile(row_x, col_y)
                pass

    def remove_tile_from_adjacencies(self, row_x, col_y, adj_list):
        neighbours, _ = self.get_neighbours(self.road_sets, row_x, col_y, is_4_neighbourhood=True)
        
        # print(neighbours)
        neighbours = set(neighbours.flatten().tolist())
        # print(neighbours)
        #remove any invalid neighbours
        try:
            neighbours.remove(0)
        except KeyError:
            pass
        #give you the road network ids
        for neighbour in neighbours:
            adj_list[neighbour].remove((row_x, col_y))

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
            road_adj_dict[neighbour].add((row_x, col_y))

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

        neighbours, center = self.get_neighbours(self.road_sets, row_x, col_y, is_4_neighbourhood=True)
        road_local_coordinates = np.argwhere(neighbours>0)
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
            self.road_graph.add_node((row_x, col_y))

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

            self.road_graph.add_node((row_x, col_y))
            #add edge betweeo all existing road tiles
            road_coordinates = np.array([row_x, col_y]) - (center - road_local_coordinates)
            for road_coordinate in road_coordinates:
                self.road_graph.add_edge((row_x, col_y), (int(road_coordinate[0]), int(road_coordinate[1])))

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
            
            #cache connectivity
            self.road_connected_sets = [node_set for node_set in nx.connected_components(self.road_graph)]

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
    
    def remove_road_tile(self, row_x, col_y):
        #get road id
        old_road_id = int(self.road_sets[row_x, col_y])
        
        #remove road tile from graph
        self.road_graph.remove_node((row_x, col_y))
        #check connected components. If connected components +1
        new_road_connected_sets = [node_set for node_set in nx.connected_components(self.road_graph)]
        
        relabel_coordinates = {}
        original_set = None

        for old_set in self.road_connected_sets:
            #connected components are mutually exclusive
            #so basically the only component must include the deleted road_tile
            #and there can only be one.
            if (row_x, col_y) in old_set:
                original_set = old_set
                break
        
        new_sets = []
        # print("Check connected components", len(new_road_connected_sets), len(self.road_connected_sets))

        # check to see which set is affected
        # if deleted node  break a connected component into other smaller  
        # components, each smaller component _must_ be a subset of the larger component
        # even if it does not break, new set is definitely smaller
        # regardless unrelated components should still remain the same
        for new_set in new_road_connected_sets:
            if new_set < original_set:
                new_sets.append(new_set)

        #the biggest set gets to keep the original road id        
        new_sets.sort(key=lambda x: len(x), reverse=True)
        # print("new sets to see", new_sets)
        for i, new_set in enumerate(new_sets):
            new_set_row = []; new_set_col = []
            for x, y in new_set:
                new_set_row.append(x)
                new_set_col.append(y)

            if i==0:
                #biggest component gets to keep their old set
                relabel_coordinates[old_road_id] = (new_set_row, new_set_col)
            else:
                #this set needs a new label
                #in the case where there is no disconnected component, 
                #it will never reach here
                relabel_coordinates[self.road_set_id] = (new_set_row, new_set_col)
                self.road_set_id+=1

        #modify the road_set
        for new_road_id, (rows, cols) in relabel_coordinates.items():
            if new_road_id==old_road_id:
                #these are already labeled
                continue
            self.road_sets[rows, cols] = new_road_id
        
        # check if industry, service, residence tiles are affected
        # they are affected in 2 ways:
        #   1) the deleted road tile is the only tile that is connecting them to a network
        #   2) they now belong to fragmented networks
        # (1) and (2) are not mutually exclusive. 
        # screw this, we just relabel everything related, solve (1) and (2) at the same time!
        
        self.road_sets[row_x, col_y] = 0
        self.road_connected_sets = new_road_connected_sets

        #check various layers and make sure that they are not affected
        self.relabel_adjacencies(old_road_id, relabel_coordinates, self.residence_tiles, self.road_adj_to_residence)
        self.relabel_adjacencies(old_road_id, relabel_coordinates, self.industry_tiles, self.road_adj_to_industries)
        self.relabel_adjacencies(old_road_id, relabel_coordinates, self.service_tiles, self.road_adj_to_services)

    def relabel_adjacencies(self, old_road_id, relabel_coordinates, tiles_of_interest, adj_list):
        #delete the entries related to the old road id
        if old_road_id in adj_list:
            del adj_list[old_road_id]
        # print("Relabeling")
        # for new_road_id, (rows, cols) in relabel_coordinates.items():
        #     for r, c in zip(rows, cols):
        #         print(f"{new_road_id}: ({r}, {c})")

        for new_road_id, (rows, cols) in relabel_coordinates.items():
            
            #basically, get all direct neighbours from all road_cells related to the network
            mask = np.zeros(self.road_sets.shape)
            mask[rows, cols]=1
            st = generate_binary_structure(2,1)
            mask = grey_dilation(mask, footprint = iterate_structure(st,1), mode='constant')
            tiles_affected = tiles_of_interest * mask

            #get the coordinates and assoicate them to the new network
            tiles_coordinates = np.argwhere(tiles_affected>0)
            if len(tiles_coordinates)>0:
                #sanity check
                adj_list[new_road_id] = set([])
            for coordinate in tiles_coordinates:
                adj_list[new_road_id].add((int(coordinate[0]), int(coordinate[1])))