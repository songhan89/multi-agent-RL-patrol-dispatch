import gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from utils.helper import *
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete


class PatrolEnv(MultiAgentEnv):
    def __init__(self, config: EnvContext):

        self._schedule_attr = ['time', 'incident', 'patrol_penalty', 'patrol_util']
        self._agent_ids = set(config['agent_ids'])
        self._scenarios = config['scenarios']
        self._subsector_map = config['subsectors_map']
        self._agents_map = config['agents_map']
        self._initial_schedules = config['initial_schedules']
        self._sectors = list(self._initial_schedules.keys())
        self._travel_matrix = config['travel_matrix']
        self._num_subsectors = len(self._travel_matrix.keys())
        self._num_agents = len(self._agent_ids)
        self._T_max = config['T_max']
        self.dones = set()
        self._spaces_in_preferred_format = True
        self._timestep = 0
        self._schedule_state = np.zeros(shape=(self._num_subsectors + 1, self._num_agents, self._T_max), dtype=int)
        self._agent_status = {}
        #set to true if an incident has been responded to
        self._responded = False

        state_dict, action_dict = {}, {}
        for agent in self._agent_ids:
            # state_dict[agent] = Tuple((
            #                             Box(low=0, high=self._num_subsectors + 1,
            #                               shape=(self._num_agents, self._T_max),
            #                               dtype=np.int32),
            #                            Discrete(self._T_max + 1), # timestep
            #                            Discrete(self._num_subsectors + 2), #incident location
            #                            Discrete(2), #responded before or not
            #                            MultiDiscrete([2, self._num_subsectors, self._T_max]), #agent status
            #                            Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
            # ))
            state_dict[agent] = gym.spaces.Tuple((
                    gym.spaces.Box(low=0, high=10, shape=(3, 72), dtype=np.int32),
                    gym.spaces.Box(low=0, high=10, shape=(3, 5), dtype=np.float),
                ))
            action_dict[agent] = gym.spaces.Discrete(2)
            self._agent_status[agent] = {'status': None, 'dest': None, 'arrival_time': None}

        self.observation_space = gym.spaces.Dict(state_dict)
        self.action_space = gym.spaces.Dict(action_dict)

        super().__init__()


    def reset(self):

        self._responded = False
        schedule = -1 * np.ones(shape=(self._num_agents, self._T_max))
        first_scenario = self._scenarios[0]
        incident_idx = first_scenario.get_location()
        t_incident = first_scenario.get_start_time()

        res = {}
        for agent in self._agent_ids:
            res[agent] = self.observation_space[agent].sample()
        self.dones = {agent: False for agent in self._agent_ids}
        return res

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i] = self.observation_space[i].sample()
            rew[i] = 1.0
            done[i] = False
            info[i] = {}

        done["__all__"] = len(self.dones) == len(self._agent_ids)
        return obs, rew, done, info

    def seed(self, seed=None):
        random.seed(seed)




