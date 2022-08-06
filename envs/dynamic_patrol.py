import gym
import numpy as np
import random
from collections import OrderedDict
from constants.Settings import NUM_DISPATCH_ACTIONS
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
        self._all_jobs = set(self._travel_matrix.keys())
        self._incidents = self._create_scenarios()

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
            action_dict[agent] = gym.spaces.Discrete(NUM_DISPATCH_ACTIONS)

        self.observation_space = gym.spaces.Dict(state_dict)
        self.action_space = gym.spaces.Dict(action_dict)

        super().__init__()


    def reset(self):

        #reinitialize schedule object
        self._state = Schedule(self._T_max, self._num_agents, self._subsector_map, self._agents_map)
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

        #set initial schedule in Schedule obj
        self._state.update_state(initial_schedule=res_schedule)

        #time step of incident
        incident_idx = self._subsector_map['subsector_2_idx'][first_scenario.get_location().get_id()]
        t_incident = first_scenario.get_start_time() // 10

        #update timestep of incident, subsector location of incident and response status
        self._state.update_state(time_step=[t_incident], incident_loc=[incident_idx],
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

        self.dones = set()

        #set timestep to t_incident
        self._timestep = t_incident

        return res

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        state_schedule = self._state.get_state('schedule')
        incident_loc_idx = self._state.get_state('incident_loc')
        agent_travel_status = self._state.get_state('agent_travel_status')
        # print (action_dict.items())
        # print (f"Timestamp: {self._timestep}")
        # print("Before--------------------")
        # self._state.to_string()
        # print ("After--------------------")

        for agent_id, action in action_dict.items():
            agent_idx = self._agents_map['agentid_2_idx'][agent_id]

            #if respond/dispatch to incident
            if action == 0:
                self._action_respond(agent_id)
            #if continue same action
            elif action == 1:
                self._action_continue(agent_id)
            # travel to nearest old locations
            elif action == 2:
                self._action_travel_nearest(agent_id, method='old')
            #travel to nearest new locations
            elif action == 3:
                self._action_travel_nearest(agent_id, method='new')

            # if agent is travelling
            is_agent_travelling = bool(agent_travel_status[agent_idx])
            # did incident occur at this timestep
            is_incident = True if incident_loc_idx != -1 else False

            rew[agent_id] = self._reward(agent_id, action, is_agent_travelling, is_incident)

            info[agent_id] = {}

        self._timestep += 1

        #set incident for next timestep
        incident = self._incidents.get(self._timestep, None)
        if incident is None:
            incident_loc_idx = - 1
        else:
            incident_loc = incident['subsector_id']
            incident_loc_idx = self._subsector_map['subsector_2_idx'][incident_loc]

        self._state.update_state(incident_loc=[incident_loc_idx])
        self._state.update_state(time_step=[self._timestep])

        for agent_id in self._agent_ids:
            obs[agent_id] = self._state.get_observation()
            if self._timestep > self._T_max:
                done[agent_id] = True
                self.dones.add(agent_id)
            else:
                done[agent_id] = False

        done["__all__"] = len(self.dones) == len(self._agent_ids)

        # if done["__all__"]:
        #     print ("--------End of Episode-------")
        #     self._state.to_string()
        #     print("------------------------------\n")

        return obs, rew, done, info

    def seed(self, seed=None):
        random.seed(seed)

    def _action_respond(self, agent_id):
        pass

    def _action_continue(self, agent_id):

        t = self._timestep
        agent_idx = self._agents_map['agentid_2_idx'][agent_id]
        travel_status_state = self._state.get_state('agent_travel_status')
        arrival_time_state = self._state.get_state('agent_arrival_time')
        schedule_state = self._state.get_state('schedule')
        travel_dest_state = self._state.get_state('agent_travel_dest')

        #if agent was patrolling
        if travel_status_state[agent_idx] == 0:
            #continue to patrol the same area
            schedule_state[agent_idx][t] = travel_dest_state[agent_idx]
            arrival_time_state[agent_idx] = t
        #if agent was travelling
        elif travel_status_state[agent_idx] == 1:
            #if still travelling, continue travelling
            if t < arrival_time_state[agent_idx]:
                schedule_state[agent_idx][t] = -1
            else:
                #reach destination
                schedule_state[agent_idx][t] = travel_dest_state[agent_idx]
                arrival_time_state[agent_idx] = t
                #change to patrol
                travel_status_state[agent_idx] = 0

        self._state.update_state(agent_arrival_time=arrival_time_state, schedule=schedule_state,
                                 agent_travel_status=travel_status_state)

    def _action_travel_nearest(self, agent_id, method='old'):

        t = self._timestep
        agent_idx = self._agents_map['agentid_2_idx'][agent_id]
        travel_status_state = self._state.get_state('agent_travel_status')

        if travel_status_state[agent_idx] == 1:
            self._action_continue(agent_id)
            return

        arrival_time_state = self._state.get_state('agent_arrival_time')
        schedule_state = self._state.get_state('schedule')
        travel_dest_state = self._state.get_state('agent_travel_dest')
        agent_cur_loc_idx = schedule_state[agent_idx][t-1]
        agent_cur_loc = self._subsector_map['idx_2_subsector'][agent_cur_loc_idx]

        old_jobs = set()
        for idx in range(len(self._agent_ids)):
            if travel_status_state[idx] == 0:
                old_jobs.add(self._subsector_map['idx_2_subsector'][travel_dest_state[idx]])
            elif travel_status_state[idx] == 1:
                t_lookback = t - 1
                while t_lookback >= 0 and schedule_state[idx][t_lookback] == -1:
                    t_lookback -= 1
                origin_idx = schedule_state[idx][t_lookback]
                old_jobs.add(self._subsector_map['idx_2_subsector'][origin_idx])

        new_jobs = self._all_jobs - old_jobs

        if method == 'new':
            job_list = list(new_jobs - {agent_cur_loc})
        elif method == 'old':
            job_list = list(old_jobs - {agent_cur_loc})

        min_job_dist = 1e7
        nearest_job_loc = []

        if len(job_list) == 0:
            job_list = list(new_jobs)

        for job_loc in job_list:
            job_dist = self._travel_matrix[agent_cur_loc][job_loc]
            if job_dist <= min_job_dist:
                nearest_job_loc.append(job_loc)
                min_job_dist = job_dist

        chosen_job_loc = random.choice(nearest_job_loc)

        schedule_state[agent_idx][t] = -1

        travel_status_state[agent_idx] = 1
        arrival_time_state[agent_idx] = t + min_job_dist
        travel_dest_state[agent_idx] = self._subsector_map['subsector_2_idx'][chosen_job_loc]

        self._state.update_state(agent_arrival_time=arrival_time_state, schedule=schedule_state,
                                 agent_travel_status=travel_status_state, agent_travel_dest=travel_dest_state)


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
                #0 for patrolling
                travel_status[agent_idx] = 0
                travel_dest[agent_idx] = time_table[agent_idx][t]
                travel_arrival[agent_idx] = t-1

        return travel_status, travel_dest, travel_arrival

    def _create_scenarios(self):

        incident_list = {}

        for incident in self._scenarios:
            incident_subsector = incident.get_location().get_id()
            t_start = incident.get_start_time() // 10
            t_end = t_start + incident.get_resolution_time() // 10
            incident_list[t_start] = {'subsector_id': incident_subsector, 'start': t_start, 'end': t_end}

        return incident_list

    def _reward(self, agent_id, action, is_agent_travelling, is_incident):

        if action == 0:
            if not is_agent_travelling and is_incident:
                return 1.0
            else:
                return -1e7

        if action == 1:
            return 1.0

        if action in [2,3]:
            if is_agent_travelling:
                return -1e7
            else:
                return 1.0





