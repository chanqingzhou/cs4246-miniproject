import gym
from gym.utils import seeding
from gym_grid_driving.envs.grid_driving import LaneSpec

def construct_random_lane_env():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-3, -1]), 
                        LaneSpec(cars=8, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=7, speed_range=[-3, -1]), 
                        LaneSpec(cars=8, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=7, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=8, speed_range=[-3, -1])],
               'random_lane_speed': True
            }
    return gym.make('GridDriving-v0', **config)