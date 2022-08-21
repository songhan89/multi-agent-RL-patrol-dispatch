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
from ray import tune
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
    return f"learned"
    #return f"policy_{agent_id}"

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    default_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    parser = argparse.ArgumentParser(description="Police Patrol environment")
    parser.add_argument("--sectors", default='A', type=str)
    parser.add_argument("--model", default="PPO", type=str)
    parser.add_argument("--max_iter", type=int, default=100, help="Number of iterations to train.")
    parser.add_argument("--poisson_mean", default=2, type=int)
    parser.add_argument("--encoding_size", default=5, type=int)
    parser.add_argument("--checkpoint", default='./checkpoint/')
    parser.add_argument("--reward_policy", default='stepwise', choices=['stepwise', 'end_of_episode'], type=str)
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--num_envs_per_worker", default=1, type=int)
    parser.add_argument("--num_workers", default=3, type=int)

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
        #TODO: Include more than 1 incident training scenario. Would it break the code ?
        training_scenarios.extend(generate_scenario(sectors[sector_id], 1, args.poisson_mean)[0])
        for patrol_area in sectors[sector_id].get_all_patrol_areas():
            subsectors_map['subsector_2_idx'][patrol_area.get_id()] = idx
            subsectors_map['idx_2_subsector'][idx] = patrol_area.get_id()
            idx += 1

    for i, src in enumerate(subsectors_map['subsector_2_idx'].keys()):
        time_matrix[src] = {}
        for j, dest in enumerate(subsectors_map['subsector_2_idx'].keys()):
            #convert to time unit
            if src == dest:
                dist_unit = 0
            else:
                dist_unit = max(1, global_time_matrix[src][dest] // 10)
            time_matrix[src][dest] = dist_unit

    #Minimum patrol requirement
    Q_j = get_global_Q_j(sectors)

    with open('./config/config.json', 'r') as f:
        config_json = json.load(f)

    initial_schedules = {}

    agents_ids = []
    for sector_id in args.sectors:
        initial_schedules[sector_id] = []
        for file in glob.glob(f'./data/Training/{args.poisson_mean}/Sector_{sector_id}/initial_schedule_*.pkl'):
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

    #TODO: Check why obs_space and action_space can't be inferred from the env?
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

    # agent_policy = {}
    # for k in agents_map['agentid_2_idx'].keys():
    #     key = f'policy_{k}'
    #     key = f'policy_shared'
    #     agent_policy[key] = PolicySpec(observation_space=obs_space,
    #                                 action_space=action_space)

    ray.init()

    print ("----------------------------------------")
    print ("Starting Rlib training")
    print (f"Sectors: {args.sectors}")
    print (f"Number of agents: {num_agents}")
    print (f"Number of subsectors: {num_subsectors}")
    print("----------------------------------------")

    experiment_name = ""
    for key, value in vars(args).items():
        experiment_name += f"{value}_"

    tune.run(
        args.model,
        name=experiment_name,
        stop={
            "training_iteration": args.max_iter
        },
        local_dir=args.checkpoint,
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        config={
            "env": PatrolEnv,
            "num_gpus": args.num_gpus,
            "num_envs_per_worker": args.num_envs_per_worker,
            "num_workers": args.num_workers,
            "env_config": {'travel_matrix': time_matrix,
                           'agent_ids': agents_ids,
                           'T_max': T_max,
                           'initial_schedules': initial_schedules,
                           'subsectors_map': subsectors_map,
                           'agents_map': agents_map,
                           'scenarios': training_scenarios,
                           'Q_j': Q_j,
                           'reward_policy': args.reward_policy
                           },  # config to pass to env class

            "multiagent": {
                "policies_to_train": ["learned"],
                "policies": {
                    "learned": PolicySpec(observation_space=obs_space,
                                          action_space=action_space)
                },
                "policy_mapping_fn": policy_mapping_fn
            },
            "preprocessor_pref": "rllib",
            "gamma": 0.99,
            "lr": 0.0001,
            "explore": True,
            "exploration_config": {
                "type": "StochasticSampling",
                "random_timesteps": 0,  # timesteps at beginning, over which to act uniformly randomly
            },
        },
    )

    # trainer = ppo.PPOTrainer(env=PatrolEnv, config={
    #     # "num_envs_per_worker": 1,
    #     # "num_workers": 6,
    #     "env_config": {'travel_matrix': time_matrix,
    #                   'agent_ids': agents_ids,
    #                   'T_max': T_max,
    #                   'initial_schedules': initial_schedules,
    #                   'subsectors_map': subsectors_map,
    #                   'agents_map': agents_map,
    #                   'scenarios': training_scenarios,
    #                   'Q_j': Q_j,
    #                   'reward_policy': args.reward_policy
    #                   },  # config to pass to env class
    #
    #     "multiagent": {
    #         "policies_to_train": ["learned"],
    #         "policies": {
    #             "learned": PolicySpec(observation_space=obs_space,
    #                                 action_space=action_space)
    #         },
    #         "policy_mapping_fn": policy_mapping_fn
    #     },
    #     "preprocessor_pref": "rllib",
    #     "gamma": 0.99,
    #     "lr": 0.0001,
    #     "explore": True,
    #     "exploration_config": {
    #         "type": "StochasticSampling",
    #         "random_timesteps": 0,  # timesteps at beginning, over which to act uniformly randomly
    #     },
    #     # "model": {
    #     #     "fcnet_hiddens": [256, 512, 256, 128],
    #     #     "fcnet_activation": "relu",
    #     #     "conv_filters": None,
    #     #     "conv_activation": "relu",
    #     #     "use_attention": True,
    #     #
    #     # }
    #
    # })
    #
    # while True:
    #     print(trainer.train())

if __name__ == "__main__":
    main()






