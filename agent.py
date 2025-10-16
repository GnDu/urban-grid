import mesa
import random

BARREN=0
RESIDENCE=1
GREENERY=2
INDUSTRY=3
SERVICE=4
ROAD=5

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
        Reset internal variables, save them as needed
        """
        raise NotImplementedError("This is an abstract class, subclass and implement it")
    
    #the two actions

    def place(self, x, y, tile):
        #check if x, y is even applicable 
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
