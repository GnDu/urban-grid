import mesa
import random
import utils

class CityPlanner(mesa.Agent):
    """
    Abstract class for agent in SIM city 
    """

    def __init__(self, model):
        super().__init__(model)


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
    
    def start_new_trial(self, **kwargs):
        """
        Reset internal variables, but not any policy. Carry them forward
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")
    
    #the two actions

    def place(self, x, y, tile):
        #check if tile is even applicable. Throw an error if that is the case
        #should check for any illegal action before hand
        self.model.grid.tile
        pass

    def destroy(self, x, y):
        #revert the tile to a barren
        #if x,y is not barren, add a fix poll_g to total_pollution
        pass
    

class RandomPlanner(CityPlanner):

    def __init__(self, model):
        super().__init__(model)

    def decide(self):
        #just randomly pick a random tile and goooo
        self.model.width
        self.model.height
