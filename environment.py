import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import PropertyLayer

class CityModel(mesa.model):

    def __init__(self, agent_class, width, height, seed=None):
        super().__init__(seed=seed)
        agent_class.create_agents(model=self, n=1)
        
        self.width = width
        self.height = height
        
        self.reset()

    def step(self):
        self.agents.do("decide")

    def reset(self):
        #reset all environment
        
        self.total_population = 0
        self.total_pollution = 0
        self.population_cap = 0

        self.grid = OrthogonalMooreGrid([self.width, self.height], random=self.random)

        pop_g_layer = PropertyLayer(
            "pop_g", (self.width, self.height), default_value=0, dtype=float
        )
        poll_g_layer = PropertyLayer(
            "poll_g", (self.width, self.height), default_value=0, dtype=float
        )

        self.grid.add_property_layer(pop_g_layer)
        self.grid.add_property_layer(poll_g_layer)

    def update_pop_and_poll(self):
        #run through the grid and update total population and pollution
        pass