import os
import csv
import sys
import random
import gym
import ray
import glob
import argparse
import logging
import json
import pickle
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from envs.dynamic_patrol import PatrolEnv
from util.ScheduleUtil import get_global_Q_j
from data.ScenarioGenerator import generate_scenario
from constants.Settings import NUM_DISPATCH_ACTIONS, T
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune import Callback
from ray.air import session

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # agent0 -> main0
    # agent1 -> main1
    return f"learned"
    # return f"policy_{agent_id}"

class ResultSave(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Trials: {trial}")
        print(f"Iteration: {iteration}")
        #print(f"Keys: {result.keys()}")
        if len(result['custom_metrics']) > 0:
            perc_incidents_responded_mean = result['custom_metrics']['perc_incidents_responded_mean']
            final_obj_val = result['custom_metrics']['final_schedule_obj_val_mean']
            initial_obj_val = result['custom_metrics']['initial_schedule_obj_val_mean']
            hamming_dist = result['custom_metrics']['hamming_dist_mean']
        else:
            perc_incidents_responded_mean = None
            final_obj_val = None
            initial_obj_val = None
            hamming_dist = None
        print(f"episodes_total: {result['episodes_total']}")
        print(f"training_iteration: {result['training_iteration']}")
        print(f"time_total: {result['time_total_s']}")
        print(f"episodes_this_iter: {result['episodes_this_iter']}")
        print(f"perc_incidents_responded_mean: {perc_incidents_responded_mean}")
        print(f"episode_reward_mean: {result['episode_reward_mean']}")
        print(f"Final obj value: {final_obj_val}")
        print(f"hamming_distance: {hamming_dist}")

        data = [result['episodes_total'],
                result['training_iteration'],
                result['time_total_s'],
                result['episode_reward_mean'],
                perc_incidents_responded_mean,
                final_obj_val,
                hamming_dist
                ]
        with open(f"./result/{trial}.csv", 'a') as f:
            writer = csv.writer(f) #this is the writer object
            writer.writerow(data) #this is the data


class MetricCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        # if worker.policy_config["batch_mode"] == "truncate_episodes":
        #     # Make sure this episode is really done.
        #     assert episode.batch_builder.policy_collectors["default_policy"].batches[
        #         -1
        #     ]["dones"][-1], (
        #         "ERROR: `on_episode_end()` should only be called "
        #         "after episode is done!"
        #     )
        episode.custom_metrics["perc_incidents_responded"] = None
        episode.custom_metrics["final_schedule_obj_val"] = None
        episode.custom_metrics["initial_schedule_obj_val"] = None
        episode.custom_metrics["hamming_dist"] = None
        #episode.custom_metrics["save_fpath"] = None

        for agent_id in episode.get_agents():
            if len(episode.last_info_for(agent_id)) > 0:
                episode.custom_metrics["perc_incidents_responded"] = \
                    episode.last_info_for(agent_id)["perc_incidents_responded"]
                episode.custom_metrics["final_schedule_obj_val"] = \
                    episode.last_info_for(agent_id)["final_schedule_obj_val"]
                episode.custom_metrics["initial_schedule_obj_val"] = \
                    episode.last_info_for(agent_id)["initial_schedule_obj_val"]
                episode.custom_metrics["hamming_dist"] = \
                    episode.last_info_for(agent_id)["hamming_dist"]



def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    default_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    parser = argparse.ArgumentParser(description="Police Patrol environment")
    parser.add_argument("--sectors", default='A', type=str)
    parser.add_argument("--model", default="PPO", type=str)
    parser.add_argument("--max_iter", type=int, default=10000, help="Number of iterations to train.")
    parser.add_argument("--poisson_mean", default=2, type=int)
    parser.add_argument("--encoding_size", default=5, type=int)
    parser.add_argument("--checkpoint", default='./checkpoint/')
    parser.add_argument("--result", default='./result/')
    parser.add_argument("--reward_policy", default='stepwise', choices=['stepwise', 'end_of_episode'], type=str)
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--num_envs_per_worker", default=1, type=int)
    parser.add_argument("--num_workers", default=3, type=int)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--resume", default=False, type=eval, choices=[True, False])
    parser.add_argument("--exploration", default='StochasticSampling', type=str,
                        choices=['StochasticSampling', 'EpsilonGreedy'])
    parser.add_argument("--benchmark",  default=False, type=eval, choices=[True, False])
    parser.add_argument("--num_scenario", default=50, type=int)

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
    for i in range(args.num_scenario):
        scenario = []
        for sector_id in args.sectors:
            sectors[sector_id] = data.get_master_table()[sector_id]
            #TODO: Include more than 1 incident training scenario. Would it break the code ?
            scenario.extend(generate_scenario(sectors[sector_id], 1, args.poisson_mean)[0])

        #create scenario if it is not a benchmark
        if not args.benchmark:
            training_scenarios.append(scenario)

    #setup subsector mapping
    for sector_id in args.sectors:
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

    exploration_config = config_json['exploration_config'][args.exploration]

    initial_schedules = {}

    agents_ids = []

    #if not running benchmark training, then load initial schedules from archive
    if not args.benchmark:
        for sector_id in args.sectors:
            initial_schedules[sector_id] = []
            for file in glob.glob(f'./data/Training/{args.poisson_mean}/Sector_{sector_id}/initial_schedule_*.pkl'):
                with open(file, 'rb') as f:
                    schedule = pickle.load(f)
                    initial_schedules[sector_id].append(schedule.get_time_tables())
            #agent ids
            agents_ids.extend(list(initial_schedules[sector_id][0].keys()))
    #load Waldy's training instance if otherwise
    else:
        print ("Load Waldy's training instance")
        with open(f'./data/training_instances_{args.sectors}.pkl', 'rb') as f:
            training_instance = pickle.load(f)
            for sector_id in args.sectors:
                initial_schedules[sector_id] = []
                for idx in range(args.num_scenario): #args.num_scenario
                    schedule = training_instance[idx][0]
                    #add initial schedule list
                    initial_schedules[sector_id].append(schedule[sector_id].get_time_tables())
                    #add scenario list
                agents_ids.extend(list(initial_schedules[sector_id][0].keys()))

            for idx in range(args.num_scenario):
                idx = 0
                schedule = training_instance[idx][0]
                training_scenarios.append(training_instance[idx][1][0])

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
    # i = 0
    # for k in agents_map['agentid_2_idx'].keys():
    #     key = f'policy_{k}'
    #     #key = f'policy_shared'
    #     agent_policy[key] = PolicySpec(observation_space=obs_space,
    #                                 action_space=action_space,
    #                                    config={"agent_id": i})
    #     i += 1

    ray.init()

    print ("----------------------------------------")
    print ("Starting Rlib training")
    print (f"Sectors: {args.sectors}")
    print (f"Number of agents: {num_agents}")
    print (f"{agents_map}")
    print (f"Number of subsectors: {num_subsectors}")
    print("----------------------------------------")

    #set experiment name - for resume to reuse checkpoint
    experiment_name = f"{args.sectors}_{args.model}_{args.max_iter}_{args.poisson_mean}_{args.encoding_size}"\
                      f"_{args.exploration}"

    #if it is not a resume, delete the checkpoint and start over

    checkpoint_path = Path(args.checkpoint, experiment_name)

    if checkpoint_path.exists() and checkpoint_path.is_dir():
        if args.resume:
            print(f"Previous checkpoint {checkpoint_path} exists. Reloading from this checkpoint.")
        else:
            print(f"Previous checkpoint {checkpoint_path} exists. Overwriting this checkpoint.")
            shutil.rmtree(checkpoint_path)
    else:
        print(f"No checkpoint {checkpoint_path} exists.")

    #output for metric result
    cur_date = datetime.now().strftime("%Y%m%d_%H%M")
    prefix_fpath = f"{cur_date}_{experiment_name}"

    if args.resume:
        print ("Restoring from previous checkpoint")
        tuner = tune.Tuner.restore(
            path=Path().resolve().joinpath(checkpoint_path).as_posix(),
            resume_unfinished=True,
            resume_errored=True,
            restart_errored=True
        )
    else:
        tuner = tune.Tuner(
            args.model,
            run_config=air.RunConfig(
                name=experiment_name,
                stop={
                "training_iteration": args.max_iter
                },
                local_dir=args.checkpoint,
                sync_config=tune.SyncConfig(
                    syncer=None  # Disable syncing
                ),
                verbose=args.verbose,
                callbacks=[ResultSave()],
                log_to_file=f"prefix_fpath.log"
                # resume=args.resume,
                # checkpoint_freq=args.max_iter // 100,
                # checkpoint_at_end=True,
                # max_failures=3,
            ),
            param_space={
                "env": PatrolEnv,
                "callbacks": MetricCallback,
                "num_gpus": args.num_gpus,
                "num_envs_per_worker": args.num_envs_per_worker,
                "num_workers": args.num_workers,
                "log_level": "WARN",
                "env_config": {'travel_matrix': time_matrix,
                               'agent_ids': agents_ids,
                               'T_max': T_max,
                               'initial_schedules': initial_schedules,
                               'subsectors_map': subsectors_map,
                               'agents_map': agents_map,
                               'scenarios': training_scenarios,
                               'Q_j': Q_j,
                               'reward_policy': args.reward_policy,
                               'prefix_fpath': experiment_name,
                               },  # config to pass to env class
                # "multiagent": {
                #     "policies": agent_policy,
                #     "policy_mapping_fn": policy_mapping_fn
                # },
                "multiagent": {
                    "policies_to_train": ["learned"],
                    "policies": {
                        "learned": PolicySpec(observation_space=obs_space,
                                              action_space=action_space)
                    },
                    "policy_mapping_fn": policy_mapping_fn
                },
                "preprocessor_pref": "rllib",
                "gamma": 1.0,
                "lr": 0.0001, #0.0001
                "explore": True,
                "exploration_config": exploration_config,
                "ignore_worker_failures": True,
                "recreate_failed_workers": True,
                "model": {
                    # "fcnet_hiddens": [64, 64],
                    # "fcnet_activation": "relu",
                    # "use_attention": False,
                    "conv_filters": None,
                    "conv_activation": "relu",
                }
            },
        )
    result = tuner.fit().get_best_result()

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






