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
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy
from envs.dynamic_patrol import PatrolEnv
from util.ScheduleUtil import get_global_Q_j
from data.ScenarioGenerator import generate_scenario
from ray.rllib.agents import ppo, qmix

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
        time_matrix[src] = {}
        for j, dest in enumerate(subsectors_map['subsector_2_idx'].keys()):
            time_matrix[src][dest] = global_time_matrix[src][dest]

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

    for idx, agent_id in enumerate(agents_ids):
        agents_map['agentid_2_idx'][agent_id] = idx
        agents_map['idx_2_agentid'][idx] = agent_id

    # print (env.step({'kml_4': 0, 'kml_2': 1, 'kml_15': 0}))
    tune.register_env(
        "ExampleEnv",
        lambda c: PatrolEnv()
    )

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # agent0 -> main0
        # agent1 -> main1
        return f"main{agent_id[-1]}"

    ray.init(local_mode=True)

    obs_space = gym.spaces.Tuple((
                    gym.spaces.Box(low=0, high=10, shape=(3, 72), dtype=np.int32),
                    gym.spaces.Box(low=0, high=10, shape=(3, 5), dtype=np.float),
                ))
    action_space = gym.spaces.Discrete(2)

    trainer = ppo.PPOTrainer(env=PatrolEnv, config={
        "env_config": {'travel_matrix': time_matrix,
                      'agent_ids': agents_ids,
                      'T_max': 72,
                      'initial_schedules': initial_schedules,
                      'subsectors_map': subsectors_map,
                      'agents_map': agents_map,
                      'scenarios': training_scenarios
                      },  # config to pass to env class
        "multiagent": {
            "policies": {
                "main4": PolicySpec(observation_space=obs_space,#gym.spaces.Discrete(2),
                                    action_space=action_space),
                "main2": PolicySpec(observation_space=obs_space,#gym.spaces.Discrete(2),
                                    action_space=action_space),
                "main5": PolicySpec(observation_space=obs_space,#gym.spaces.Discrete(2),
                                     action_space=action_space)
            },
            "policy_mapping_fn": policy_mapping_fn
        }
    })

    # tune.run(
    #     "PPO",
    #     stop={"episode_reward_mean": 200},
    #     config={
    #         "env": "ExampleEnv",
    #         "num_gpus": 0,
    #         "num_workers": 1,
    #         "multiagent": {
    #             "policies": {
    #                 "kml_4": PolicySpec(observation_space=obs_space,
    #                                     action_space=action_space),
    #                 "kml_2": PolicySpec(observation_space=obs_space,
    #                                     action_space=action_space),
    #                 "kml_15": PolicySpec(observation_space=obs_space,
    #                                     action_space=action_space),
    #                 "random": PolicySpec(policy_class=RandomPolicy),
    #             },
    #             "policy_mapping_fn": policy_mapping_fn,
    #             "policies_to_train": ["kml_4"]
    #         },
    #         "framework": "torch"
    #     }
    # )

    while True:
        print(trainer.train())

if __name__ == "__main__":
    main()






