import os
import csv
import sys
import random
import gym
import pandas as pd
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
from constants.Settings import NUM_DISPATCH_ACTIONS, T, MAX_HAMMING_DISTANCE
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune import Callback
from ray.air import session
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.appo import APPO
from ray.air.checkpoint import Checkpoint
from ray.rllib.algorithms.algorithm import Algorithm


logging_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
logging.basicConfig(filename=os.path.join(logging_dir, 'madprp.log'),
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"policy_{agent_id}"

def get_policy(policy, agents_map, obs_space, action_space):

    if policy == 'single':
        return {
            "policies_to_train": ["learned"],
            "policies": {
                "learned": PolicySpec(observation_space=obs_space,
                                        action_space=action_space)
            },
            "policy_mapping_fn": lambda x: 'learned'
        }

    elif policy == 'multi':
        agent_policy = {}
        for i, k in enumerate(agents_map['agentid_2_idx'].keys()):
            key = f'policy_{k}'
            agent_policy[key] = PolicySpec(observation_space=obs_space,
                                        action_space=action_space,
                                           config={"agent_id": i})
        return {
            "policies": agent_policy,
            "policy_mapping_fn": policy_mapping_fn
        }

class ResultSave(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        logger.debug(f"Trials: {trial}")
        logger.debug(f"Iteration: {iteration}")
        if len(result['custom_metrics']) > 0:
            perc_incidents_responded_mean = result['custom_metrics']['perc_incidents_responded_mean']
            final_obj_val = result['custom_metrics']['final_schedule_obj_val_mean']
            initial_obj_val = result['custom_metrics']['initial_schedule_obj_val_mean']
            hamming_dist = result['custom_metrics']['hamming_dist_mean']
            perc_incidents_responded_myopic = result['custom_metrics']['perc_incidents_responded_myopic_mean']
            benchmark_obj_val = result['custom_metrics']['benchmark_obj_val_mean']
        else:
            perc_incidents_responded_mean = None
            final_obj_val = None
            initial_obj_val = None
            hamming_dist = None
            perc_incidents_responded_myopic = None
            benchmark_obj_val = None

        logger.debug(f"episodes_total: {result['episodes_total']}")
        logger.debug(f"training_iteration: {result['training_iteration']}")
        logger.debug(f"time_total: {result['time_total_s']}")
        logger.debug(f"episodes_this_iter: {result['episodes_this_iter']}")
        logger.debug(f"perc_incidents_responded_mean: {perc_incidents_responded_mean}")
        logger.debug(f"episode_reward_mean: {result['episode_reward_mean']}")
        logger.debug(f"Final obj value: {final_obj_val}")
        logger.debug(f"hamming_distance: {hamming_dist}")

        data = [result['episodes_total'],
                result['training_iteration'],
                result['time_total_s'],
                result['episode_reward_mean'],
                perc_incidents_responded_mean,
                final_obj_val,
                hamming_dist,
                perc_incidents_responded_myopic,
                benchmark_obj_val
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
        episode.custom_metrics["perc_incidents_responded"] = None
        episode.custom_metrics["final_schedule_obj_val"] = None
        episode.custom_metrics["initial_schedule_obj_val"] = None
        episode.custom_metrics["hamming_dist"] = None
        episode.custom_metrics["perc_incidents_responded_myopic"] = None
        episode.custom_metrics["benchmark_obj_val"] = None
        episode.custom_metrics["worst_response"] = None


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
                episode.custom_metrics["perc_incidents_responded_myopic"] = \
                    episode.last_info_for(agent_id)["perc_incidents_responded_myopic"]
                episode.custom_metrics["benchmark_obj_val"] = \
                    episode.last_info_for(agent_id)["benchmark_obj_val"]
                episode.custom_metrics["worst_response"] = \
                    episode.last_info_for(agent_id)["worst_response"]

def main():
    parser = argparse.ArgumentParser(description="Police Patrol environment")
    parser.add_argument("--test", default=False, type=eval, choices=[True, False], help='Run evaluation on test set')
    parser.add_argument("--sectors", default='EFL', type=str, help='Training sectors (e.g EFL means sectors E,F,L)')
    parser.add_argument("--model", default="PPO", type=str, help='APPO, PPO')
    parser.add_argument("--max_iter", type=int, default=500, help="Number of iterations to train.")
    parser.add_argument("--poisson_mean", default=2, type=int,
                        help="Param by Poisson distribution for generating incidents.")
    parser.add_argument("--encoding_size", default=5, type=int,
                        help="Size of neural network encoder")
    parser.add_argument("--checkpoint", default='./checkpoint/', help="Output path for saving checkpoint")
    parser.add_argument("--result", default='./result/', help="Result output")
    parser.add_argument("--reward_policy", default='end_of_episode',
                        choices=['stepwise', 'end_of_episode'], type=str,
                        help="Stepwise = reward on patrol presence given at each immediate step. "
                             "End of episode = reward given at end of episode")
    parser.add_argument("--checkpoint_frequency", default=5, type=int, help="Model checkpoint frequency")
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--num_envs_per_worker", default=1, type=int)
    parser.add_argument("--num_workers", default=3, type=int, help="Number of workers for parallel sampling")
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--resume", default=False, type=eval, choices=[True, False],
                        help="Resume training from checkpoint")
    parser.add_argument("--exploration", default='Greedy', type=str,
                        choices=['StochasticSampling', 'EpsilonGreedy', 'Greedy'])
    parser.add_argument("--benchmark",  default=False, type=eval, choices=[True, False],
                        help='to run training with myopic benchmark dataset')
    parser.add_argument("--num_scenario", default=500, type=int,
                        help='Number of initial schedules/scenarios for training')
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--policy", default='single', choices=['single', 'multi'], type=str,
                        help="Multi means each agent has its own policy network. Single policy is shared by all agents.")
    parser.add_argument("--nn_network", default='default', choices=['default', '128x128_relu_attention'],
                        type=str, help="Size of neural network.")
    parser.add_argument("--theta_step", default=1.0, type=float, help='Penalty coefficient for Hamming distance.')

    args = parser.parse_args()

    logger.info("Load input data..." + "\n")
    # Load a preprocessed data
    with open("./data/processed_data.pkl", "rb") as fp:
        data = pickle.load(fp)

    global_time_matrix = data.get_time_matrix()
    time_matrix = {}
    sectors = {}
    sectors_info = {}
    subsectors_map = {'subsector_2_idx':{}, 'idx_2_subsector':{},
                      'subsector_parent_sector': {}, 'idx_parent_sector': {}}
    agents_map = {'agentid_2_idx':{}, 'idx_2_agentid':{}}
    training_scenarios = []
    training_benchmark = []

    #get all subsectors under each sector into a global list
    idx = 0
    for i in range(args.num_scenario):
        scenario = []
        for sector_id in args.sectors:
            sectors[sector_id] = data.get_master_table()[sector_id]
            scenario.extend(generate_scenario(sectors[sector_id], 1, args.poisson_mean)[0])

        #create scenario if it is not a benchmark
        if not args.benchmark:
            training_scenarios.append(scenario)
            training_benchmark.append((0, 0))

    #setup subsector mapping
    for sector_id in args.sectors:
        #append all subsectors under the sector to this info dict
        sectors_info[sector_id] = list(sectors[sector_id].get_patrol_areas_table().keys())
        for patrol_area in sectors[sector_id].get_all_patrol_areas():
            subsectors_map['subsector_parent_sector'][patrol_area.get_id()] = sector_id
            subsectors_map['subsector_2_idx'][patrol_area.get_id()] = idx
            subsectors_map['idx_2_subsector'][idx] = patrol_area.get_id()
            subsectors_map['idx_parent_sector'][idx] = sector_id
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
    model_config = config_json['model'][args.nn_network]
    initial_schedules = {}
    agents_ids = []
    initial_schedules_test = {}
    agents_ids_test = []
    num_test_scenarios = len(glob.glob(f'./data/Testing/{args.poisson_mean}/Sector_{sector_id}/initial_schedule_*.pkl'))

    #Load test data set
    if args.test:
        for sector_id in args.sectors:
            initial_schedules_test[sector_id] = []
            for file in glob.glob(f'./data/Testing/{args.poisson_mean}/Sector_{sector_id}/initial_schedule_*.pkl'):
                with open(file, 'rb') as f:
                    test_schedule = pickle.load(f)
                    initial_schedules_test[sector_id].append(test_schedule.get_time_tables())
            #agent ids
            agents_ids_test.extend(list(initial_schedules_test[sector_id][0].keys()))

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
        logger.info ("Load Waldy's training instance")
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
                training_scenarios.append(training_instance[idx][1][0])
                #response ratio and obj value
                training_benchmark.append((training_instance[idx][2][1]/training_instance[idx][2][-1],
                                           training_instance[idx][2][2]))

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

    #set experiment name - for resume to reuse checkpoint
    experiment_name = f"{args.sectors}_{args.model}_{args.max_iter}_{args.poisson_mean}_{args.encoding_size}" \
                      f"_{args.num_scenario}_{args.theta_step}_{args.exploration}_{args.policy}_{args.nn_network}"

    logger.info ("----------------------------------------")
    logger.info ("Starting Rlib training")
    logger.info(f"Sectors: {experiment_name}")
    logger.info (f"Sectors: {args.sectors}")
    logger.info (f"Number of agents: {num_agents}")
    logger.info (f"{agents_map}")
    logger.info (f"Number of subsectors: {num_subsectors}")
    logger.info("----------------------------------------")
    ray.init()

    #if it is not a resume, delete the checkpoint and start over
    checkpoint_path = Path(args.checkpoint, experiment_name)

    if checkpoint_path.exists() and checkpoint_path.is_dir():
        if args.resume or args.test:
            logger.info(f"Previous checkpoint {checkpoint_path} exists. Reloading from this checkpoint.")
        else:
            logger.info(f"Previous checkpoint {checkpoint_path} exists. Overwriting this checkpoint.")
            shutil.rmtree(checkpoint_path)
    else:
        logger.info(f"No checkpoint {checkpoint_path} exists.")

    #output for metric result
    cur_date = datetime.now().strftime("%Y%m%d_%H%M")
    prefix_fpath = f"{cur_date}_{experiment_name}"

    #create checkpoint folder
    result_path = Path().resolve().joinpath(checkpoint_path)
    result_path.mkdir(parents=True, exist_ok=True)

    if args.test:
        logger.info("Running evaluation on test set. Restoring from previous model checkpoint")

        test_env = PatrolEnv({'travel_matrix': time_matrix,
                               'agent_ids': agents_ids_test,
                               'T_max': T_max,
                               'initial_schedules': initial_schedules_test,
                               'subsectors_map': subsectors_map,
                               'agents_map': agents_map,
                               'scenarios': training_scenarios,
                               'Q_j': Q_j,
                               'reward_policy': args.reward_policy,
                               'benchmark': training_benchmark,
                               'sectors_info': sectors_info,
                               'theta_step': args.theta_step,
                               })

        trainer = APPO(config={
                "env": PatrolEnv,
                "framework": "tf",
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
                               'benchmark': training_benchmark,
                               'sectors_info': sectors_info,
                               'theta_step': args.theta_step,
                               },
                "preprocessor_pref": "rllib",
                "multiagent": get_policy(args.policy, agents_map, obs_space, action_space),
                "preprocessor_pref": "rllib",
                "gamma": 1.0,
                "lr": args.learning_rate,  # 0.0001
                "explore": True,
                "exploration_config": exploration_config,
                "ignore_worker_failures": True,
                "recreate_failed_workers": True,
                "model": model_config,
            }
        )

        #if args.resume:
        print (f"Restoring model from checkpoint {checkpoint_path}")
        trainer.restore(
            Path().resolve().joinpath(checkpoint_path, 'best_checkpoint'))

        # run until all test scenarios are evaluated
        print (f"Running evaluation on test data set.")
        i = 0
        test_metrics_df = pd.DataFrame()
        while i < num_test_scenarios:
            done = False
            obs = test_env.reset()
            action = {}
            while not done:
                for agent_id, agent_obs in obs.items():
                    policy_id = f"policy_{agent_id}"
                    action[agent_id] = trainer.compute_single_action(agent_obs, policy_id=policy_id)
                obs, reward, dones, info = test_env.step(action)
                done = dones['__all__']
            for k, v in info.items():
                if len(v) > 0:
                    test_metrics_df = test_metrics_df.append(v, ignore_index=True)
            i += 1

        test_metrics_df.to_csv(Path().resolve().joinpath(checkpoint_path, 'test_metrics.csv'), index=False)

    else: #Run for training

        checkpoint_config = air.CheckpointConfig(
            checkpoint_frequency=args.checkpoint_frequency,
            num_to_keep=1,
            checkpoint_score_order="max",
        )

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
                checkpoint_config=checkpoint_config,
                verbose=args.verbose,
                # callbacks=[ResultSave()], #not required anymore since it is saved in `progress.csv`
                log_to_file=f"{prefix_fpath}.log"
            ),
            param_space={
                "env": PatrolEnv,
                "framework": "tf",
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
                               'benchmark': training_benchmark,
                               'sectors_info': sectors_info,
                               'theta_step': args.theta_step,
                               },  # config to pass to env class
                "multiagent": get_policy(args.policy, agents_map, obs_space, action_space),
                "preprocessor_pref": "rllib",
                "gamma": 1.0,
                "lr": args.learning_rate,  # 0.0001
                "explore": True,
                "exploration_config": exploration_config,
                "ignore_worker_failures": True,
                "recreate_failed_workers": True,
                "model": model_config,
            },
        )

        # Resume training if checkpoint exists
        # TODO: Unclear why this is not always working, best to let training start from scratch and complete whole
        # There seems to be issue loading the model from checkpoint
        restore_path = Path().resolve().joinpath(checkpoint_path).as_posix()
        if args.resume:
            logger.info ("Restoring from previous checkpoint")
            tuner.restore(
                path=restore_path,
                resume_unfinished=True,
                resume_errored=True,
                restart_errored=True
            )
        # Start training from scratch
        else:
            logger.info ("Starting training from scratch")

        # Save the best model checkpoint thus far based on % of incidents responded to
        result = tuner.fit().get_best_result(metric='mean_perc_incidents_responded', mode='max', scope='last-10-avg')
        model_checkpoint = result.checkpoint
        _ = model_checkpoint.to_directory(Path().resolve().joinpath(checkpoint_path, 'best_checkpoint').as_posix())

if __name__ == "__main__":
    main()





