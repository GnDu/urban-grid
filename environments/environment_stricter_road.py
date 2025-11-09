import numpy as np
from collections import defaultdict
from scipy.ndimage import  grey_dilation, \
                            generate_binary_structure, iterate_structure


from environments.environment import CityModel
from utils import TileTypes

class CityModelStricterRoad(CityModel):

    def __init__(self, agent_class, width, height, update_rules, init_road_tile, collect_rate = 1.0, max_cluster_size=-1, seed=None):
        # super().__init__(agent_class=agent_class, 
        #                     width=width, 
        #                     height=height, 
        #                     update_rules=update_rules, 
        #                     collect_rate=collect_rate, 
        #                     seed=seed)
        super().__init__(agent_class, width, height, update_rules, collect_rate, seed)
        #for cluster
        self.max_cluster_size = max_cluster_size #currently not being used
        self.global_cluster_id = 1
        self.residence_cluster = np.zeros((self.width, self.height), dtype=np.int64)
        self.greenery_cluster = np.zeros((self.width, self.height), dtype=np.int64)
        self.industry_cluster = np.zeros((self.width, self.height), dtype=np.int64)
        self.service_cluster = np.zeros((self.width, self.height), dtype=np.int64)
        
        #clusters that are adjacent to road
        self.cluster_adjacent_to_road = set([])
        #mapping cluster id to co-ordindates
        self.cluster_co_ordinates = defaultdict(set)

        #set the road tile to (0,7) and update the road network
        self.grid.tile._mesa_data[init_road_tile] = TileTypes.ROAD.value
        self.book_keep()

    @staticmethod
    def get_valid_intial_road_tiles(rows, cols):
        valid_init_roads = [(0,0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
        for r in range(1, rows-1):
            valid_init_roads.append((r,0))
            valid_init_roads.append((r,cols-1))
        for c in range(1, cols-1):
            valid_init_roads.append((0,c))
            valid_init_roads.append((rows-1,c))

        return valid_init_roads

    @staticmethod
    def get_random_init_road_tile(rows, cols, seed=None):
        from random import Random
        rand = Random(seed)
        return rand.choice(CityModelStricterRoad.get_valid_intial_road_tiles(rows, cols))
        
    def set_tile(self, row_x, col_y, tile):
        
        #raise error when road is not part of a network
        if tile==TileTypes.ROAD.value:
            legal_tiles = self.get_legal_road_tiles()
            if  not (legal_tiles==[row_x, col_y]).all(axis=1).any():
                raise RuntimeError("Invalid road placements for ROAD tile!")

        #because mesa has a really convulted way to set discrete_space property directly
        #so we are just going to access its private attribute directly.
        self.grid.tile._mesa_data[row_x, col_y] = tile

        if tile==TileTypes.ROAD.value:
            self.update_road_connected_cluster(row_x, col_y)
        elif tile==TileTypes.RESIDENCE.value:
            self.update_cluster(row_x, col_y, self.residence_cluster)
        elif tile==TileTypes.INDUSTRY.value:
            self.update_cluster(row_x, col_y, self.industry_cluster)
        elif tile==TileTypes.SERVICE.value:
            self.update_cluster(row_x, col_y, self.service_cluster)
        elif tile==TileTypes.GREENERY.value:
            self.update_cluster(row_x, col_y, self.greenery_cluster)
        elif tile==TileTypes.BARREN.value:
            #we don't allow removing of any BARREN tile
            raise RuntimeError("Not allowed to remove any tiles now")

    def get_legal_road_tiles(self):
        """
        get legal road tiles please use this to get which road tiles are good

        Returns:
            co_ordinates: legal tiles that can set as road tiles
        """
        st = generate_binary_structure(2,1)
        legal_tiles = grey_dilation(self.road_tiles, footprint = iterate_structure(st,1), mode='constant')
        legal_tiles-=self.road_tiles
        return np.argwhere(legal_tiles>0)


    def update_cluster(self, row_x, col_y, cluster_set):
        neighbours, center = self.get_neighbours(cluster_set, row_x, col_y, is_4_neighbourhood=True)
        neighbours = set(neighbours.flatten().tolist())
        #cluster id must be >0
        #remove any invalid ids
        try:
            neighbours.remove(0)
        except KeyError:
            pass
        
        curr_cluster_id = None
        if len(neighbours)==0:
            #if there are no neighbours, create a new cluster id
            curr_cluster_id = self.global_cluster_id
            cluster_set[row_x, col_y] = curr_cluster_id
            self.global_cluster_id+=1
        else:
            #if there are more than one cluster, merge them.
            #the smaller cluster should be subsumed under the bigger one

            #determine who has the bigger network
            max_count = -1
            max_neighbour = -1
            for neighbour in neighbours:
                total_in_network = (cluster_set==neighbour).sum()
                if total_in_network > max_count:
                    max_count = total_in_network
                    max_neighbour = neighbour

            #set the new tile to the one with the biggest network
            cluster_set[row_x, col_y] = max_neighbour

            # self.road_graph.add_node((row_x, col_y))
            # #add edge betweeo all existing road tiles
            # cluster_coordinates = np.array([row_x, col_y]) - (center - cluster_local_coordinates)
            # for cluster_coordinate in cluster_coordinates:
            #     self.road_graph.add_edge((row_x, col_y), (int(cluster_coordinate[0]), int(cluster_coordinate[1])))

            curr_cluster_id = max_neighbour
            #now set the rest of the neighour network to that max_neighbour network
            for neighbour in neighbours:
                if neighbour==max_neighbour:
                    continue
                cluster_set[cluster_set==neighbour] = max_neighbour

                #if the neighbour was connected to a road, remove them
                if neighbour in self.cluster_adjacent_to_road:
                    self.cluster_adjacent_to_road.remove(neighbour)
                    #but add the new max_neighbour in
                    self.cluster_adjacent_to_road.add(max_neighbour)
                
                #mapping of cluster to co-ordinates need to be updated as well.
                self.cluster_co_ordinates[max_neighbour]|=self.cluster_co_ordinates[neighbour]
                #remove old neighbour
                del self.cluster_co_ordinates[neighbour]
            
        #now check and see if a road is adjacent to a cluster
        #check road neighbour
         #if there are raod, this cluster is connected to it
        neighbours, center = self.get_neighbours(self.road_tiles, row_x, col_y, is_4_neighbourhood=True)
        neighbours = set(neighbours.flatten().tolist())
        try:
            neighbours.remove(0)
        except KeyError:
            pass

        if neighbours:
            self.cluster_adjacent_to_road.add(curr_cluster_id)


    def update_road_connected_cluster(self, row_x, col_y):
        # simplyfying this even more
        # just need to check if a newly extended road connects to any cluster

        #sanity check
        assert(int(self.grid.tile._mesa_data[(row_x, col_y)])==TileTypes.ROAD.value)

        cluster_to_check = [self.residence_cluster, self.greenery_cluster, self.industry_cluster, self.service_cluster]
        for cluster in cluster_to_check:
            neighbours, center = self.get_neighbours(cluster, row_x, col_y, is_4_neighbourhood=True)
            neighbours = set(neighbours.flatten().tolist())
            try:
                neighbours.remove(0)
            except KeyError:
                pass

            #neighbours will be a list of cluster ids
            for neighbour in neighbours:
                self.cluster_adjacent_to_road.add(neighbour)