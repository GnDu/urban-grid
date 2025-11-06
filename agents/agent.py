import mesa
from utils import TileTypes

class CityPlanner(mesa.Agent):
    """
    Abstract class for agent in SIM city 
    """

    def __init__(self, model, destroy_tile_penalty:float=10, **kwargs):
        super().__init__(model)
        self.destroy_tile_penalty = destroy_tile_penalty
        self.total_population = 0
        self.total_pollution = 0
        self.population_cap = 0

    def decide(self):
        """
        How the agent decide what step to take next
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")


    def update(self, **kwargs):
        """
        How the agent update any internal state
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")
    
    def warm_start(self, **kwargs):
        """
        Initializing a new agent with knowledge, policies, or parameters learned from previous episodes or tasks
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")
    
    #the two actions

    def place(self, row_x, col_y, tile):
        #check if tile is even applicable. Throw an error if that is the case
        #should check for any illegal action before hand
        #note, tile should not be BARREN. That's destroy tile
        assert(tile!=TileTypes.BARREN.value)
        x_y_tile = self.model.get_tile(row_x, col_y)
        if x_y_tile!=TileTypes.BARREN.value:
            raise RuntimeError(f"({row_x}, {col_y}): {x_y_tile} is not BARREN")
        self.model.set_tile(row_x, col_y, tile)
        
    def destroy(self, row_x, col_y):
        #revert the tile to a barren
        #if x,y is not barren, add a fix poll_g to total_pollution
        self.model.set_tile(row_x, col_y, TileTypes.BARREN.value)
        #increment total polution
        self.total_pollution+=self.destroy_tile_penalty
    

class RandomPlanner(CityPlanner):

    def __init__(self, model):
        super().__init__(model)

    def decide(self):
        #just randomly pick a random tile and goooo
        self.model.width
        self.model.height
