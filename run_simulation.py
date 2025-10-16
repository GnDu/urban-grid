import argparse
import importlib

import environment
from environment import CityModel
import agent

def get_class(classname):
    #classname should be in the form of module.classname
    #Example: agent.RandomPlanner
    module, class_name= classname.split('.',1)
    module = importlib.import_module(module)
    return getattr(module, class_name)

def instantiate_model(agent_classname, model_classname, seed):
    agent_class = get_class(agent_classname)
    model_class = get_class(model_classname)
    model = model_class(agent_class, seed=None)
    return model

def run(model, steps):

    for i in range(steps):
        agent = model.agents[0]
        #let the agent decide 
        model.step()

        #calculate population gain (pop_g) and pollution gain (poll_g)
        agent.update_pop_and_poll()
        
        #update any internal states, like utiity, etc
        agent.update()

def main(agent_classname, steps, seed=None):
    
    model = instantiate_model(agent_classname, seed)
    
    run(model, steps)


if __name__=="__main__":
    args = argparse.ArgumentParser()

    #do any intialistion here
    main()