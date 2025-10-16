import mesa
from mesa.discrete_space import OrthogonalMooreGrid

class CityModel(mesa.model):

    def __init__(self, agent_class, width, height, seed=None):
        super().__init__(seed=seed)
        agent_class.create_agents(model=self, n=1)
        
        self.width = width
        self.height = height
        
        self.total_population = 0
        self.total_pollution = 0
        self.population_cap = 0

        self.grid = OrthogonalMooreGrid([width, height], random=self.random)

    def step(self):
        self.agents.do("decide")

    
    def update_pop_and_poll(self):
        #run through the grid and update total population and pollution
        pass