# SIMple City (SIM City)

## Introduction

Modern urban planning often struggles to balance residential expansion with rising industrial pollution. This project aims to develop a simple AI-based planning model that optimizes land use on a small grid by placing residential and industrial zones while minimizing pollution exposure for residents. The system will explore how different layouts affect the overall livability of the city, finding an optimal balance between accommodating population growth and controlling pollution within limited space.

To that end, enters SIM City: a simple city building game. The game involves the agent to place city tiles on a grid so as to maximise population growth. However, most city tiles contribute to pollution, so the agent will to need to minimise this.

Rationality: population of a city can be co-related with the prosperity / attraction of a city. The more populous a city is, the more attractive the city is, but it also brings increasing population. So to simplify things, population is a good proxy for the prosperity of a city. Thus, sustainability can be defined as _maximising_ prosperity while _minimising_ pollution.

## Goal

The goal of the game is maximise total population while minimising total pollution accrued within a specified duration.

## Project Structure

```
.
├── agents/
│   └── agent.py
├── data/
│   ├── outputs/
│   └── update_parameters/
│       └── DefaultUpdateRule.json
├── environments/
│   └── environment.py
├── update_rules/
│   └── update_rules.py
├── city_sim_rules.md               // game/simulation rules
├── example_run.ipynb               //a notebook on how to interface with the environment
├── README.md
├── utils.py                        //any utility function shared across all modules
└── visualisation.py                //helper function for visulisation
```

I highly recommend extending the base classes in each python files for development:

- If creating new agent, extend it from the `CityPlanner` class in `agent.py`.
- If extending functionalities of environment, extend it from `CityModel` class in `environment.py`
- If creating new update rules, extend it from `DefaultUpdateRules` in `update_rules.py`

## Requirements

This project is developed on python 3.12. The packages are defined in requirements.txt. Note, the lack of pytorch modules and what not, this will definitely change.

## How it works

Please refer to `example_run.ipynb` for a quick walkthrough. You can also look at `environment_example.ipynb` to check how the environment works.

If you're looking for a breakdown of the game mechanics, refer to the `city_sim_rules.md`.

## TODO

Goals:

- We can also tweak the goal to be hitting $x$ total population while making sure the agent does not hit $y$ pollution.

Environment:

- ~~Industry and Service tiles should be connected to a Resident tile via Road tiles before they are 'activated'~~
- Finding the 'proper' values for pollution, population gain/loss, pollution mitigations, and any modifiers.

Visualisation:

- Mesa use Solaris to create a web application out of the simulation. This allows users to visualise the data in real time as the simulation runs. I have not integrated a way to do this yet.
- Fix the animation play button.