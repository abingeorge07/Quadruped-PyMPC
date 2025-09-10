from enum import Enum
import numpy as np

# # Import config 
# import config as cfg

class GaitType(Enum):
    """Enumeration class to represent the different gaits that a quadruped robot can perform."""

    TROT = 0
    PACE = 1
    BOUNDING = 2
    CIRCULARCRAWL = 3
    BFDIAGONALCRAWL = 4
    BACKDIAGONALCRAWL = 5
    FRONTDIAGONALCRAWL = 6
    FULL_STANCE = 7


# Gait phase definitions for each gait type
gaits = {
    'crawl':  {'FL': 0.0, 'FR': 0.5, 'RL': 0.75, 'RR': 0.25},
    # 'walk':  {'FL': 0.50, 'FR': 0.0, 'RL': 0.75, 'RR': 0.25},
    'trot':  {'FL': 0.0, 'FR': 0.5, 'RL': 0.5,  'RR': 0.0},
    'pace':  {'FL': 0.0, 'FR': 0.0, 'RL': 0.5,  'RR': 0.5},
    'bound': {'FL': 0.0, 'FR': 0.0, 'RL': 0.5,  'RR': 0.5},
    'full_stance': {'FL': 0.0, 'FR': 0.0, 'RL': 0.0,  'RR': 0.0},
}


# Gait phase helper function
def get_gait_phase(sim_param):
    """
    Get the gait phase based on the simulation parameters.

    Args:
        sim_param (dict): Simulation parameters containing gait type and parameters.

    Returns:
        dict: A dictionary with the gait phase for each leg.
    """

    # Get gait name
    gait_type = sim_param['gait']

    try:
        gait_req = gaits[gait_type]
        return gait_req
    except KeyError:
        raise ValueError(f"Unsupported gait type: {gait_type}")
    






