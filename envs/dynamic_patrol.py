import gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from utils.helper import *
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete
from state.schedule import Schedule

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
        self._num_initial_schedules = len(self._initial_schedules[self._sectors[0]])
        self._num_agents = len(self._agent_ids)
        self._T_max = config['T_max']
        self.dones = set()
        self._spaces_in_preferred_format = True
        self._timestep = 0
        #Initialize a Schedule object to keep track of the state
        self._state = Schedule(self._T_max, self._num_agents, self._subsector_map, self._agents_map)

        state_dict, action_dict = {}, {}
        for agent in self._agent_ids:
            state_dict[agent] = Tuple((
                                        Box(low=-1, high=self._num_subsectors,
                                          shape=(self._num_agents, self._T_max + 1),
                                          dtype=np.int32),
                                        #time step
                                        Box(low=0, high=self._T_max, shape=(1,), dtype=np.int32),
                                        #incident occur at which sector
                                        Box(low=-1, high=self._num_subsectors, shape=(1,), dtype=np.int32),
                                        #responded or not
                                        Box(low=0, high=1, shape=(1,), dtype=np.int32),
                                        #agent travels status (0 for patrol, 1 for travel)
                                        Box(low=0, high=1, shape=(self._num_agents,), dtype=np.int32),
                                        #agent's travel destination (set to src dest if not travelling)
                                        Box(low=-1, high=self._num_subsectors, shape=(self._num_agents,),
                                            dtype=np.int32),
                                        #timestep to arrive at the dest if agent was travelling
                                        Box(low=-1, high=self._T_max, shape=(self._num_agents,), dtype=np.int32)
            ))
            action_dict[agent] = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Dict(state_dict)
        self.action_space = gym.spaces.Dict(action_dict)

        super().__init__()


    def reset(self):

        res_schedule = self._state.get_state('schedule')
        first_scenario = self._scenarios[0]

        #randomly select an initial schedule
        selected_idx = random.randint(0, self._num_initial_schedules - 1)

        for sector in self._initial_schedules.keys():
            for agent in self._initial_schedules[sector][selected_idx].keys():
                agent_time_table = self._initial_schedules[sector][selected_idx][agent]
                agent_idx = self._agents_map['agentid_2_idx'][agent]
                res_schedule[agent_idx] = [self._subsector_map['subsector_2_idx'].get(sub_sector, -1)
                                                for sub_sector in agent_time_table]

        #time step of incident
        incident_idx = self._subsector_map['subsector_2_idx'][first_scenario.get_location().get_id()]
        t_incident = first_scenario.get_start_time()

        #update timestep of incident, subsector location of incident and response status
        self._state.update_state(incident_time=[t_incident], incident_loc=[incident_idx],
                                  responded=[0])
        #agent travel status
        res_travel, res_agent_dest, res_agent_arr = self._get_travel_status(res_schedule, t_incident)
        self._state.update_state(agent_travel_status=res_travel, agent_travel_dest=res_agent_dest,
                                  agent_arrival_time=res_agent_arr)
        #set timestep of t >= T(incident) to -1
        self._state.update_state(schedule=empty_timetable(res_schedule, t_incident))

        res = {}
        for agent in self._agent_ids:
            res[agent] = self._state.get_observation()

        self.dones = {agent: False for agent in self._agent_ids}

        return res

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        self._state.to_string()

        for i, action in action_dict.items():
            obs[i] = self.observation_space[i].sample()
            rew[i] = 1.0
            done[i] = False
            info[i] = {}

        done["__all__"] = len(self.dones) == len(self._agent_ids)
        return obs, rew, done, info

    def seed(self, seed=None):
        random.seed(seed)

    def _get_travel_status(self, time_table, t):
        """
        Return the travel status of all agents

        :param time_table:
        :param t:
        :return: Tuple of travel_status, travel_dest, travel_arrival arrays
        """

        travel_status = [0] * self._num_agents
        travel_dest = [-1] * self._num_agents
        travel_arrival = [-1] * self._num_agents

        for agent_idx in range(self._num_agents):
            if time_table[agent_idx][t] == -1:
                timestep = t + 1
                while timestep < self._T_max and time_table[agent_idx][timestep] == -1:
                    timestep += 1

                if timestep == self._T_max:
                    #1 for travel
                    travel_status[agent_idx] = 1
                    travel_dest[agent_idx] = -1
                    travel_arrival[agent_idx] -1
                else:
                    travel_status[agent_idx] = 1
                    travel_dest[agent_idx] = time_table[agent_idx][timestep]
                    travel_arrival[agent_idx] = timestep
            else:
                travel_status[agent_idx] = 0
                travel_dest[agent_idx] = time_table[agent_idx][t]
                travel_arrival[agent_idx] = t

        return travel_status, travel_dest, travel_arrival




