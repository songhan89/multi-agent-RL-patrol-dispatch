import os
import sys
import random
import gym
import ray
import glob
import argparse
import logging
import json
import pickle
import numpy as np
from datetime import datetime
from ray.rllib.policy.policy import PolicySpec
from envs.dynamic_patrol import PatrolEnv
from util.ScheduleUtil import get_global_Q_j
from data.ScenarioGenerator import generate_scenario
from constants.Settings import NUM_DISPATCH_ACTIONS, T
from ray.rllib.agents import ppo, qmix
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # agent0 -> main0
    # agent1 -> main1
    return f"main{agent_id[-1]}"

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    default_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    parser = argparse.ArgumentParser(description="Police Patrol environment")
    parser.add_argument("--sectors", default='A', type=str)
    parser.add_argument("--model", default="PPO", type=str)
    parser.add_argument("--episode", default=10000, type=int)
    parser.add_argument("--poisson_mean", default=2, type=int)
    parser.add_argument("--encoding_size", default=5, type=int)
    parser.add_argument("--checkpoint", default=None)

    args = parser.parse_args()

    logging.info("Load input data..." + "\n")
    # Load a preprocessed data
    with open("./data/processed_data.pkl", "rb") as fp:
        data = pickle.load(fp)

    global_time_matrix = data.get_time_matrix()
    time_matrix = {}
    sectors = {}
    subsectors_map = {'subsector_2_idx':{}, 'idx_2_subsector':{}}
    agents_map = {'agentid_2_idx':{}, 'idx_2_agentid':{}}
    training_scenarios = []

    #get all subsectors under each sector into a global list
    idx = 0
    for sector_id in args.sectors:
        sectors[sector_id] = data.get_master_table()[sector_id]
        training_scenarios.extend(generate_scenario(sectors[sector_id], 1, args.poisson_mean)[0])
        for patrol_area in sectors[sector_id].get_all_patrol_areas():
            subsectors_map['subsector_2_idx'][patrol_area.get_id()] = idx
            subsectors_map['idx_2_subsector'][idx] = patrol_area.get_id()
            idx += 1

    for i, src in enumerate(subsectors_map['subsector_2_idx'].keys()):
        shortest_dist = 1e7
        time_matrix[src] = {}
        for j, dest in enumerate(subsectors_map['subsector_2_idx'].keys()):
            #convert to time unit
            if src == dest:
                dist_unit = 0
            else:
                dist_unit = max(1, global_time_matrix[src][dest] // 10)
            time_matrix[src][dest] = dist_unit

    Q_j = get_global_Q_j(sectors)

    with open('./config/config.json', 'r') as f:
        config_json = json.load(f)

    initial_schedules = {}

    agents_ids = []
    for sector_id in args.sectors:
        initial_schedules[sector_id] = []
        for file in glob.glob(f'./data/Training/{args.poisson_mean}/Sector_{args.sectors}/initial_schedule_*.pkl'):
            with open(file, 'rb') as f:
                schedule = pickle.load(f)
                initial_schedules[sector_id].append(schedule.get_time_tables())
        #agent ids
        agents_ids.extend(list(initial_schedules[sector_id][0].keys()))

    num_agents = len(agents_ids)
    T_max = len(T) - 1
    num_subsectors = len(subsectors_map['subsector_2_idx'].keys())

    for idx, agent_id in enumerate(agents_ids):
        agents_map['agentid_2_idx'][agent_id] = idx
        agents_map['idx_2_agentid'][idx] = agent_id

    obs_space = Tuple((
                    Box(low=-1, high=num_subsectors,
                        shape=(num_agents, T_max + 1),
                        dtype=np.int32),
                    # time step
                    Box(low=0, high=T_max + 1, shape=(1,), dtype=np.int32),
                    # incident occur at which sector
                    Box(low=-1, high=num_subsectors, shape=(1,), dtype=np.int32),
                    # responded or not
                    Box(low=0, high=1, shape=(1,), dtype=np.int32),
                    # agent travels status (0 for patrol, 1 for travel)
                    Box(low=0, high=1, shape=(num_agents,), dtype=np.int32),
                    # agent's travel destination (set to src dest if not travelling)
                    Box(low=-1, high=num_subsectors, shape=(num_agents,),
                        dtype=np.int32),
                    # timestep to arrive at the dest if agent was travelling
                    Box(low=-1, high=T_max + 1, shape=(num_agents,), dtype=np.int32),
            ))
    action_space = gym.spaces.Discrete(NUM_DISPATCH_ACTIONS)

    ray.init()
    trainer = ppo.PPOTrainer(env=PatrolEnv, config={
        "env_config": {'travel_matrix': time_matrix,
                      'agent_ids': agents_ids,
                      'T_max': T_max,
                      'initial_schedules': initial_schedules,
                      'subsectors_map': subsectors_map,
                      'agents_map': agents_map,
                      'scenarios': training_scenarios,
                      },  # config to pass to env class
        "multiagent": {
            "policies": {
                "main4": PolicySpec(observation_space=obs_space,
                                    action_space=action_space),
                "main2": PolicySpec(observation_space=obs_space,
                                    action_space=action_space),
                "main5": PolicySpec(observation_space=obs_space,
                                     action_space=action_space)
            },
            "policy_mapping_fn": policy_mapping_fn
        }
    })

    while True:
        print(trainer.train())

if __name__ == "__main__":
    main()






