import gym
import numpy as np
import random
import heapq as hq
from collections import OrderedDict
from copy import deepcopy
from constants.Settings import NUM_DISPATCH_ACTIONS, MAX_NUM_DISPATCH_ACTIONS, MAX_HAMMING_DISTANCE
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from utils.helper import *
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete
from state.schedule import Schedule


class PatrolEnv(MultiAgentEnv):
    def __init__(self, config: EnvContext):

        self._schedule_attr = ['time', 'incident', 'patrol_penalty', 'patrol_util']
        #set of agent ids
        self._agent_ids = set(config['agent_ids'])
        # list of incident scenarios of training scenario
        self._scenarios_list = config['scenarios']
        self._benchmark_list = config['benchmark']
        # list of sector <-> subsector mapping
        self._sectors_info = config['sectors_info']
        # initialise with the first option
        self._scenarios = self._scenarios_list[0]
        # tuple of benchmark score for myopic (index 0 - percentage of incident responded, index 1 - obj value)
        self._benchmark = self._benchmark_list[0]
        #mapping for subsector and agent ids <-> index
        self._subsector_map = config['subsectors_map']
        self._agents_map = config['agents_map']
        #choice of reward policy
        self._reward_policy = config['reward_policy']
        #param for hamming distance penalty step func, only for `end_of_episode` reward
        self._theta_step = config['theta_step']
        #list of initial schedules
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
        #counter track for number of incidents responded in each sector
        self._responded_sectors = {k:{'count':0, 'responded':0} for k in self._sectors_info.keys()}
        self._responded = 0
        self._response_utility = 0
        #convert str keys to int keys
        #special handler for resume training since keys are loaded as str
        self._subsector_map['subsector_2_idx'] = {int(k):v for k,v in self._subsector_map['subsector_2_idx'].items()}
        self._subsector_map['idx_2_subsector'] = {int(k):v for k,v in self._subsector_map['idx_2_subsector'].items()}
        self._agents_map['idx_2_agentid'] = {int(k):v for k,v in self._agents_map['idx_2_agentid'].items()}
        self._travel_matrix = {int(k):v for k,v in self._travel_matrix.items()}
        for key, val in self._travel_matrix.items():
            self._travel_matrix[key] = {int(k): v for k, v in self._travel_matrix[key].items()}

        #Initialize a Schedule object to keep track of the state
        self._state = Schedule(self._T_max, self._num_agents, self._subsector_map, self._agents_map)
        self._all_jobs = set(self._travel_matrix.keys())
        self._incidents = self._create_scenarios()
        self._Q_j = config['Q_j']
        # special handler for resume training since keys are loaded as str
        self._Q_j = {int(k): v for k, v in self._Q_j.items()}
        self._Q_j_idx = {}
        for k, v in self._Q_j.items():
            subsector_idx = self._subsector_map['subsector_2_idx'][k]
            self._Q_j_idx[subsector_idx] = v

        state_dict, action_dict = {}, {}
        for agent in self._agent_ids:
            state_dict[agent] = Tuple((
                                        #schedule
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
        self._action_history = {}

        super().__init__()


    def reset(self):

        #reinitialize schedule object
        self._state = Schedule(self._T_max, self._num_agents, self._subsector_map, self._agents_map)
        initial_schedule = self._state.get_state('initial_schedule')
        res_schedule = self._state.get_state('schedule')
        #reset response rate at the start of the episode
        self._responded_sectors = {k:{'count':0, 'responded':0} for k in self._sectors_info.keys()}
        self._responded = 0
        self._response_utility = 0

        #reset action history
        self._action_history = {}
        for agent in self._agent_ids:
            self._action_history[agent] = {}
            for action in range(MAX_NUM_DISPATCH_ACTIONS):
                self._action_history[agent][action] = 0

        #randomly select an initial schedule
        selected_idx = random.randint(0, self._num_initial_schedules - 1)
        #TODO: This is a temporary solution to bypass incident that start at T = 0 which causes problem
        while self._scenarios_list[selected_idx][0].get_start_time() == 0:
            selected_idx = random.randint(0, self._num_initial_schedules - 1)

        # initialise scenarios
        self._scenarios = self._scenarios_list[selected_idx]
        self._benchmark = self._benchmark_list[selected_idx]
        first_scenario = self._scenarios[0]
        self._incidents = self._create_scenarios()

        for incident in self._scenarios:
            incident_subsector = incident.get_location().get_id()
            try:
                parent_sector = self._subsector_map['subsector_parent_sector'][incident_subsector]
            except:
                parent_sector = self._subsector_map['subsector_parent_sector'][str(incident_subsector)]
            self._responded_sectors[parent_sector]['count'] += 1

        #TODO: remove fixed seed
        #selected_idx = 0

        for sector in self._initial_schedules.keys():
            for agent in self._initial_schedules[sector][selected_idx].keys():
                agent_time_table = self._initial_schedules[sector][selected_idx][agent]
                agent_idx = self._agents_map['agentid_2_idx'][agent]
                res_schedule[agent_idx] = [self._subsector_map['subsector_2_idx'].get(sub_sector, -1)
                                                for sub_sector in agent_time_table]
                initial_schedule[agent_idx] = agent_time_table

        #set initial schedule in Schedule obj
        self._state.update_state(initial_schedule=deepcopy(res_schedule))

        #time step of incident
        incident_idx = self._subsector_map['subsector_2_idx'][first_scenario.get_location().get_id()]
        t_incident = first_scenario.get_start_time() // 10
        t_incident_resolution = first_scenario.get_resolution_time() // 10
        #update timestep of incident, subsector location of incident and response status
        self._state.update_state(time_step=[t_incident],
                                 incident_loc=[incident_idx],
                                 incident_t_resolution=[t_incident_resolution],
                                 responded=[0])
        #agent travel status
        res_travel, res_agent_dest, res_agent_arr = self._get_travel_status(res_schedule, t_incident)
        self._state.update_state(agent_travel_status=res_travel, agent_travel_dest=res_agent_dest,
                                  agent_arrival_time=res_agent_arr)
        #set timestep of t >= T(incident) to -1
        self._state.update_state(schedule=empty_timetable(res_schedule, t_incident))

        res = {}
        #every agent shares the same state space
        for agent in self._agent_ids:
            res[agent] = self._state.get_observation()

        self.dones = set()
        #set timestep to t_incident
        self._timestep = t_incident

        obj_val = get_objective_value_MA(self._state.get_state('initial_schedule'),
                               self._subsector_map['subsector_2_idx'].keys(),
                               self._Q_j_idx)

        # print (f"Initial obj val: {obj_val}")
        # print("Reset and Initial State--------------------")
        # print(res_travel, res_agent_dest, res_agent_arr)
        # self._state.to_string()
        # print ("\n")
        # print ("*****************************")
        return res

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        state_schedule = self._state.get_state('schedule')
        incident_loc_idx = self._state.get_state('incident_loc')[0]
        incident_t_resolution = self._state.get_state('incident_t_resolution')[0]
        agent_travel_status = self._state.get_state('agent_travel_status')
        # print (action_dict.items())
        # print (f"Timestamp: {self._timestep}")
        # print (self._action_history)
        # print("Before taking the action--------------------")
        # self._state.to_string()
        # print ("-------------------------------------------")

        #shuffle the ordering of agents taking action
        agent_action_list = list(action_dict.items())
        random.shuffle(agent_action_list)

        for agent_id, action in agent_action_list:
            agent_idx = self._agents_map['agentid_2_idx'][agent_id]

            #if respond/dispatch to incident
            if action == 0:
                rew[agent_id] = self._action_respond(agent_id, incident_loc_idx, incident_t_resolution)
            #if continue same action
            elif action == 1:
                rew[agent_id] = self._action_continue(agent_id)
            # travel to nearest old locations
            #patrol area that result in schedule with least disruption
            elif action == 2:
                rew[agent_id] = self._action_least_disruption(agent_id)
            elif action == 3:
                rew[agent_id] = self._action_obj_based_jobs(agent_id, method='old')
            elif action == 4:
                rew[agent_id] = self._action_obj_based_jobs(agent_id, method='new')

            info[agent_id] = {}

        self._timestep += 1

        #set incident for next timestep
        incident = self._incidents.get(self._timestep, None)
        if incident is None:
            incident_loc_idx = - 1
            incident_t_end = -1
        else:
            incident_loc = incident['subsector_id']
            incident_t_resolution = incident['end']
            incident_loc_idx = self._subsector_map['subsector_2_idx'][incident_loc]

        #reset to zero
        self._state.update_state(responded=[0])
        self._state.update_state(incident_loc=[incident_loc_idx])
        self._state.update_state(time_step=[self._timestep])
        self._state.update_state(incident_t_resolution=[incident_t_resolution])

        for agent_id in self._agent_ids:
            obs[agent_id] = self._state.get_observation()
            if self._timestep > self._T_max:
                done[agent_id] = True
                self.dones.add(agent_id)
            else:
                done[agent_id] = False

        done["__all__"] = len(self.dones) == len(self._agent_ids)

        if done["__all__"]:
            # print ("--------End of Episode-------")
            obj_val = get_objective_value_MA(self._state.get_state('schedule'),
                                         self._subsector_map['subsector_2_idx'].keys(),
                                         self._Q_j_idx)
            #Print final state
            # print("Final Schedule--------------------")
            # self._state.to_string()
            print(f"End of episode obj val: {obj_val}")
            print(f"Number of incidents responded to: {self._responded} / {len(self._scenarios)}, "
                  f"{100.0 * self._responded / len(self._scenarios)}")
            worst_sector_response = 1e7
            for k, v in self._responded_sectors.items():
                if v['count'] == 0:
                    response_rate = 0
                else:
                    response_rate = 100.0 * v['responded'] / v['count']
                print(f"Sector {k}: Number of incidents responded to: {v['responded']}/{v['count']}, "
                  f"{response_rate}")
                if response_rate < worst_sector_response:
                    worst_sector_response = response_rate
            print(f"Worst sector response rate: {worst_sector_response}")
            # print(f"Action space distribution: {self._action_history}")
            hamming_dist = hamming_distance(self._state.get_state('initial_schedule'),
                                            self._state.get_state('schedule'))
            print(f"Hamming distance from initial schedule: {hamming_dist}")
            print("------------------------------\n")

            #get reward at the end of episode if reward policy selected is `end_of_episode`
            if self._reward_policy == 'end_of_episode':
                for agent_id in self._agent_ids:
                    rew[agent_id] = obj_val - THETA * hamming_dist
                    #step function to further penalise the reward if it exceeds soft-constraint threshold
                    if hamming_dist > MAX_HAMMING_DISTANCE:
                        rew[agent_id] -= self._theta_step

            #TODO: Double check if this is needed
            info[agent_id] = {
                'hamming_dist': hamming_dist,
                'perc_incidents_responded': 1.0 * self._responded / len(self._scenarios),
                'perc_incidents_responded_myopic': self._benchmark[0],
                'final_schedule_obj_val': obj_val,
                'initial_schedule_obj_val': get_objective_value_MA(self._state.get_state('initial_schedule'),
                               self._subsector_map['subsector_2_idx'].keys(),
                               self._Q_j_idx),
                'benchmark_obj_val': self._benchmark[1],
                'worst_response': worst_sector_response
                }

        return obs, rew, done, info

    def seed(self, seed=None):
        random.seed(seed)

    def _action_respond(self, agent_id, incident_loc_idx, incident_t_resolution):
        """
        Get agent with `agent_id` to respond to incident
        :param agent_id: agent id
        :param incident_loc_idx: incident location index
        :param incident_t_resolution: incident resolution time
        :return: reward of action
        """

        t = self._timestep
        agent_idx = self._agents_map['agentid_2_idx'][agent_id]
        initial_state = self._state.get_state('initial_schedule')
        schedule_state = self._state.get_state('schedule')
        respond_state = self._state.get_state('responded')
        agent_travel_status = self._state.get_state('agent_travel_status')
        arrival_time_state = self._state.get_state('agent_arrival_time')
        travel_dest_state = self._state.get_state('agent_travel_dest')

        obj_val_before = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)

        # if agent's status was travelling or there was no incident
        # switch the action to `continue` action
        # TODO: action masking is required in this case actually
        if agent_travel_status[agent_idx] == 1 or incident_loc_idx == -1 or t == self._T_max:
            return self._action_continue(agent_id, is_invalid_action=True)

        # if the incident has been responded by another agent already, switch action to `least disruption`
        if respond_state[0] == 1:
            return self._action_least_disruption(agent_id)
            #TODO: check if continue or least disruption is better as the default action
            #return self._action_continue(agent_id, is_invalid_action=True)

        #if agent was patrolling, get the agent to respond to the incident
        if agent_travel_status[agent_idx] == 0:
            agent_src_loc = self._subsector_map['idx_2_subsector'][travel_dest_state[agent_idx]]
            agent_dest_loc = self._subsector_map['idx_2_subsector'][incident_loc_idx]
            response_time = self._travel_matrix[agent_src_loc][agent_dest_loc]

            #if travelling is required, update agent's travel status to travelling
            if response_time > 0:
                schedule_state[agent_idx][t] = -1
                arrival_time_state[agent_idx] = t + response_time
                agent_travel_status[agent_idx] = 1
                travel_dest_state[agent_idx] = incident_loc_idx
            else:
                schedule_state[agent_idx][t] = incident_loc_idx
                arrival_time_state[agent_idx] = t
                agent_travel_status[agent_idx] = 0
                travel_dest_state[agent_idx] = incident_loc_idx

            # change response state to True so the next agent will not respond to this incident anymore
            respond_state[0] = 1
            if t + response_time <= incident_t_resolution:
                # update response rate counter only if responded within resolution time
                self._responded += 1
                try:
                    parent_sector = self._subsector_map['subsector_parent_sector'][agent_dest_loc]
                except:
                    parent_sector = self._subsector_map['subsector_parent_sector'][str(agent_dest_loc)]
                self._responded_sectors[parent_sector]['responded'] += 1

        self._state.update_state(agent_arrival_time=arrival_time_state, schedule=schedule_state,
                                 agent_travel_status=agent_travel_status, agent_travel_dest=travel_dest_state,
                                 responded=respond_state)

        objective_val_after = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)
        penalty_val = hamming_distance(initial_state, schedule_state)

        if self._reward_policy == 'end_of_episode':
            self._response_utility += response_utility_fn(response_time)
            # len(self._scenarios)
            reward = OMEGA * response_utility_fn(response_time) / len(self._incidents.keys())
        elif self._reward_policy == 'stepwise':
            # len(self._scenarios)
            reward = OMEGA * response_utility_fn(response_time) / len(self._incidents.keys()) + \
                    objective_val_after - obj_val_before - THETA * penalty_val
            #alternative reward formulation
            # reward = OMEGA * response_utility_fn(response_time) / len(self._incidents.keys()) + \
            #          - THETA * penalty_val

        #increase action count for tracking
        self._action_history[agent_id][0] += 1

        return reward


    def _action_continue(self, agent_id, is_invalid_action=False):
        """
        Continue previous action for the agent - either continue to patrol or to travel
        :param agent_id: agent id
        :param is_invalid_action: whether this action is mapped from another invalid action
        :return: reward of action
        """

        t = self._timestep
        agent_idx = self._agents_map['agentid_2_idx'][agent_id]
        initial_state = self._state.get_state('initial_schedule')
        travel_status_state = self._state.get_state('agent_travel_status')
        arrival_time_state = self._state.get_state('agent_arrival_time')
        schedule_state = self._state.get_state('schedule')
        travel_dest_state = self._state.get_state('agent_travel_dest')
        obj_val_before = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)

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
                #change agent status to patrol since has reached destination
                travel_status_state[agent_idx] = 0

        #no need to update agent destination state since the action is to continue previous action
        self._state.update_state(agent_arrival_time=arrival_time_state, schedule=schedule_state,
                                 agent_travel_status=travel_status_state)

        objective_val_after = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)
        penalty_val = hamming_distance(initial_state, schedule_state)

        if is_invalid_action:
            reward = PHI
        else:
            if self._reward_policy == 'end_of_episode':
                reward = 0
            elif self._reward_policy == 'stepwise':
                reward = objective_val_after - obj_val_before - THETA * penalty_val

        #increase action count
        self._action_history[agent_id][1] += 1

        return reward

    def _action_travel_nearest(self, agent_id, method='old'):
        """
        Travel to one of the nearest patrol area
        :param agent_id: agent
        :param method: `old` will search for locations currently being patrolled, `new` search for new locations not
                        being patrolled
        :return: reward of action
        """

        t = self._timestep
        agent_idx = self._agents_map['agentid_2_idx'][agent_id]
        travel_status_state = self._state.get_state('agent_travel_status')

        #if agent's status was travelling, continue to travel
        if travel_status_state[agent_idx] == 1 or t == self._T_max:
            return self._action_continue(agent_id, is_invalid_action=True)

        initial_state = self._state.get_state('initial_schedule')
        arrival_time_state = self._state.get_state('agent_arrival_time')
        schedule_state = self._state.get_state('schedule')
        travel_dest_state = self._state.get_state('agent_travel_dest')
        agent_cur_loc_idx = schedule_state[agent_idx][t-1]

        try:
            agent_cur_loc = self._subsector_map['idx_2_subsector'][agent_cur_loc_idx]
        except:
            raise ValueError

        obj_val_before = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)

        old_jobs = set()
        #look through all previous destinations patrolled by all agents
        for idx in range(len(self._agent_ids)):
            #if the agent's status was patrolling, add this patrol area to list
            if travel_status_state[idx] == 0:
                old_jobs.add(self._subsector_map['idx_2_subsector'][travel_dest_state[idx]])
            #if agent was travelling, look back in time until the last patrol area
            elif travel_status_state[idx] == 1:
                #look back to the last patrol area patrolled by the agent before he started travelling
                t_lookback = t - 1
                while t_lookback >= 0 and schedule_state[idx][t_lookback] == -1:
                    t_lookback -= 1
                origin_idx = schedule_state[idx][t_lookback]
                if origin_idx == -1:
                    raise ValueError
                old_jobs.add(self._subsector_map['idx_2_subsector'][origin_idx])

        new_jobs = self._all_jobs - old_jobs

        if method == 'new':
            job_list = list(new_jobs - {agent_cur_loc})
        elif method == 'old':
            job_list = list(old_jobs - {agent_cur_loc})

        nearest_job_loc = []

        #if there are no old jobs, then use new jobs
        #TODO: Think again if this is the right approach
        if len(job_list) == 0:
            job_list = list(new_jobs)

        for job_loc in job_list:
            job_dist = self._travel_matrix[agent_cur_loc][job_loc]
            if len(nearest_job_loc) == 0:
                hq.heappush(nearest_job_loc, (job_dist, job_loc))
            else:
                if nearest_job_loc[0][0] == job_dist:
                    hq.heappush(nearest_job_loc, (job_dist, job_loc))
                elif nearest_job_loc[0][0] > job_dist:
                    #flush out all job locations
                    nearest_job_loc = []
                    hq.heappush(nearest_job_loc, (job_dist, job_loc))

        dist_list = []
        #TODO: Can we refactor this as a common submodule ?
        #choose a job among all nearest jobs which has the lowest hamming distance
        for job_dist, subsector_loc in nearest_job_loc:
            next_state = deepcopy(schedule_state)
            #if agent_cur_loc != subsector_loc:
            response_time = self._travel_matrix[agent_cur_loc][subsector_loc]
            i = 0
            while i < response_time and (t + i) < self._T_max:
                next_state[agent_idx][t+i] = -1
                i += 1

            if t + i < self._T_max:
                next_state[agent_idx][t+i] = self._subsector_map['subsector_2_idx'][subsector_loc]

            dist = hamming_distance(initial_state, next_state)

            if len(dist_list) == 0:
                hq.heappush(dist_list, (dist, subsector_loc, response_time))
            else:
                if dist_list[0][0] == dist:
                    hq.heappush(dist_list, (dist, subsector_loc, response_time))
                elif dist_list[0][0] > dist:
                    dist_list = []
                    hq.heappush(dist_list, (dist, subsector_loc, response_time))

        _, chosen_job_loc, min_job_dist = random.choice(dist_list)
        selected_dest_idx = self._subsector_map['subsector_2_idx'][chosen_job_loc]

        if min_job_dist > 0:
            schedule_state[agent_idx][t] = -1
            arrival_time_state[agent_idx] = t + min_job_dist
            travel_status_state[agent_idx] = 1
            travel_dest_state[agent_idx] = selected_dest_idx
        elif min_job_dist == 0:
            schedule_state[agent_idx][t] = selected_dest_idx
            arrival_time_state[agent_idx] = t + min_job_dist
            travel_status_state[agent_idx] = 0
            travel_dest_state[agent_idx] = selected_dest_idx
        else:
            raise ValueError

        self._state.update_state(agent_arrival_time=arrival_time_state, schedule=schedule_state,
                                 agent_travel_status=travel_status_state, agent_travel_dest=travel_dest_state)

        objective_val_after = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)
        penalty_val = hamming_distance(initial_state, schedule_state)

        if self._reward_policy == 'end_of_episode':
            reward = 0
        elif self._reward_policy == 'stepwise':
            reward = objective_val_after - obj_val_before - THETA * penalty_val

        #increase action count for tracking
        if method == 'old':
            self._action_history[agent_id][2] += 1
        elif method == 'new':
            self._action_history[agent_id][3] += 1

        return reward

    def _action_least_disruption(self, agent_id, is_invalid_action=False):
        """
        Choose the next patrol area that has the shortest hammming distance from the initial schedule

        :param agent_id: agent id
        :param is_invalid_action: whether this action is mapped from another invalid action
        :return: reward of action
        """

        t = self._timestep
        initial_schedule = self._state.get_state('initial_schedule')
        agent_idx = self._agents_map['agentid_2_idx'][agent_id]
        travel_status_state = self._state.get_state('agent_travel_status')
        arrival_time_state = self._state.get_state('agent_arrival_time')
        agent_travel_status = self._state.get_state('agent_travel_status')
        schedule_state = self._state.get_state('schedule')
        travel_dest_state = self._state.get_state('agent_travel_dest')
        obj_val_before = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)

        #if agent was travelling or we have reached the end of timeline
        if travel_status_state[agent_idx] == 1 or t == self._T_max:
            return self._action_continue(agent_id, is_invalid_action=True)

        agent_src_loc = self._subsector_map['idx_2_subsector'][travel_dest_state[agent_idx]]

        dist_list = []

        #find the next patrol area that results in lowest hamming distance deviation from initial schedule
        for subsector_loc in self._travel_matrix.keys():
            next_state = deepcopy(schedule_state)
            response_time = self._travel_matrix[agent_src_loc][subsector_loc]
            i = 0
            while i < response_time and (t + i) < self._T_max:
                next_state[agent_idx][t+i] = -1
                i += 1

            if t + i < self._T_max:
                next_state[agent_idx][t+i] = self._subsector_map['subsector_2_idx'][subsector_loc]

            dist = hamming_distance(initial_schedule, next_state)
            #TODO: should not matter since the relative distance will still be larger if take hamming distance of
            #      the entire schedule
            #dist = hamming_distance([initial_schedule[agent_idx]], [next_state[agent_idx]])

            if len(dist_list) == 0:
                hq.heappush(dist_list, (dist, subsector_loc, response_time))
            else:
                if dist_list[0][0] == dist:
                    hq.heappush(dist_list, (dist, subsector_loc, response_time))
                elif dist_list[0][0] > dist:
                    dist_list = []
                    hq.heappush(dist_list, (dist, subsector_loc, response_time))

        _, selected_dest, dest_response_time = random.choice(dist_list)
        selected_dest_idx = self._subsector_map['subsector_2_idx'][selected_dest]

        if dest_response_time > 0:
            schedule_state[agent_idx][t] = -1
            arrival_time_state[agent_idx] = t + dest_response_time
            agent_travel_status[agent_idx] = 1
            travel_dest_state[agent_idx] = selected_dest_idx
        elif dest_response_time == 0:
            schedule_state[agent_idx][t] = selected_dest_idx
            arrival_time_state[agent_idx] = t + dest_response_time
            agent_travel_status[agent_idx] = 0
            travel_dest_state[agent_idx] = selected_dest_idx
        else:
            raise ValueError

        self._state.update_state(agent_arrival_time=arrival_time_state, schedule=schedule_state,
                                 agent_travel_status=agent_travel_status, agent_travel_dest=travel_dest_state)

        objective_val_after = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)
        penalty_val = hamming_distance(initial_schedule, schedule_state)

        if is_invalid_action:
            reward = PHI
        else:
            if self._reward_policy == 'end_of_episode':
                reward = 0
            elif self._reward_policy == 'stepwise':
                reward = objective_val_after - obj_val_before - THETA * penalty_val

        #increase action count for tracking
        self._action_history[agent_id][4] += 1

        return reward

    def _action_obj_based_jobs(self, agent_id, method='old'):
        """
        Travel to the patrol area (old or new jobs) that yields the best objective function
        :param agent_id:
        :param method:
        :return:
        """

        t = self._timestep
        agent_idx = self._agents_map['agentid_2_idx'][agent_id]
        travel_status_state = self._state.get_state('agent_travel_status')

        #if agent's status was travelling, continue to travel
        if travel_status_state[agent_idx] == 1 or t == self._T_max:
            return self._action_continue(agent_id, is_invalid_action=True)

        initial_state = self._state.get_state('initial_schedule')
        arrival_time_state = self._state.get_state('agent_arrival_time')
        schedule_state = self._state.get_state('schedule')
        agent_travel_status = self._state.get_state('agent_travel_status')
        travel_dest_state = self._state.get_state('agent_travel_dest')
        agent_cur_loc_idx = schedule_state[agent_idx][t-1]

        try:
            agent_cur_loc = self._subsector_map['idx_2_subsector'][agent_cur_loc_idx]
        except:
            raise ValueError

        obj_val_before = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)

        old_jobs = set()
        #look through all previous destinations patrolled by all agents
        for idx in range(len(self._agent_ids)):
            if travel_status_state[idx] == 0:
                old_jobs.add(self._subsector_map['idx_2_subsector'][travel_dest_state[idx]])
            elif travel_status_state[idx] == 1:
                #look back to the last patrol area patrolled by the agent before he started travelling
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

        #if there are no old jobs, then use new jobs
        if len(job_list) == 0:
            job_list = list(new_jobs)

        obj_list = []
        #find the next patrol area that results in best obj value
        for subsector_loc in job_list:
            next_state = deepcopy(schedule_state)
            #if agent_cur_loc != subsector_loc:
            response_time = self._travel_matrix[agent_cur_loc][subsector_loc]
            i = 0
            while i < response_time and (t + i) < self._T_max:
                next_state[agent_idx][t+i] = -1
                i += 1

            if (t + i) < self._T_max:
                next_state[agent_idx][t+i] = self._subsector_map['subsector_2_idx'][subsector_loc]

            #multiply by -1 to change min heap to max heap
            next_state_obj = -1 * get_objective_value_MA(schedule_state,
                                                         self._subsector_map['subsector_2_idx'].keys(),
                                                         self._Q_j_idx)
            if len(obj_list) == 0:
                hq.heappush(obj_list, (next_state_obj, subsector_loc))
            else:
                if obj_list[0][0] == next_state_obj:
                    hq.heappush(obj_list, (next_state_obj, subsector_loc))
                elif obj_list[0][0] > next_state_obj:
                    obj_list = []
                    hq.heappush(obj_list, (next_state_obj, subsector_loc))

        dist_list = []
        #choose the job that has the lowest hamming distance
        for job_dist, subsector_loc in obj_list:
            next_state = deepcopy(schedule_state)
            #if agent_cur_loc != subsector_loc:
            response_time = self._travel_matrix[agent_cur_loc][subsector_loc]
            i = 0
            while i < response_time and (t + i) < self._T_max:
                next_state[agent_idx][t+i] = -1
                i += 1

            if t + i < self._T_max:
                next_state[agent_idx][t+i] = self._subsector_map['subsector_2_idx'][subsector_loc]

            dist = hamming_distance(initial_state, next_state)

            if len(dist_list) == 0:
                hq.heappush(dist_list, (dist, subsector_loc, response_time))
            else:
                if dist_list[0][0] == dist:
                    hq.heappush(dist_list, (dist, subsector_loc, response_time))
                elif dist_list[0][0] > dist:
                    dist_list = []
                    hq.heappush(dist_list, (dist, subsector_loc, response_time))

        _, selected_dest, min_job_dist = random.choice(dist_list)
        selected_dest_idx = self._subsector_map['subsector_2_idx'][selected_dest]

        if min_job_dist > 0:
            schedule_state[agent_idx][t] = -1
            arrival_time_state[agent_idx] = t + min_job_dist
            agent_travel_status[agent_idx] = 1
            travel_dest_state[agent_idx] = selected_dest_idx
        elif min_job_dist == 0:
            schedule_state[agent_idx][t] = selected_dest_idx
            arrival_time_state[agent_idx] = t + min_job_dist
            travel_status_state[agent_idx] = 0
            travel_dest_state[agent_idx] = selected_dest_idx
        else:
            raise ValueError

        self._state.update_state(agent_arrival_time=arrival_time_state, schedule=schedule_state,
                                 agent_travel_status=agent_travel_status, agent_travel_dest=travel_dest_state)

        objective_val_after = get_objective_value_MA(schedule_state, self._subsector_map['subsector_2_idx'].keys(),
                                        self._Q_j_idx)
        penalty_val = hamming_distance(initial_state, schedule_state)

        if self._reward_policy == 'end_of_episode':
            reward = 0
        elif self._reward_policy == 'stepwise':
            reward = objective_val_after - obj_val_before - THETA * penalty_val

        #increase action count for tracking
        if method == 'old':
            self._action_history[agent_id][5] += 1
        elif method == 'new':
            self._action_history[agent_id][6] += 1

        return reward

    def _get_travel_status(self, time_table, t):
        """
        Return the travel status of all agents at time step t based on the time table provided

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
                if time_table[agent_idx][t-1] != -1:
                    #0 for patrolling
                    travel_status[agent_idx] = 0
                    travel_dest[agent_idx] = time_table[agent_idx][t]
                    travel_arrival[agent_idx] = t-1
                else:
                    #1 for travelling
                    travel_status[agent_idx] = 1
                    travel_dest[agent_idx] = time_table[agent_idx][t]
                    travel_arrival[agent_idx] = t

        return travel_status, travel_dest, travel_arrival

    def _create_scenarios(self):
        """
        Generate dict of incident scenarios
        :return: dict - subsector id, incident start time and end time
        """
        incident_list = {}

        for incident in self._scenarios:
            incident_subsector = incident.get_location().get_id()
            t_start = incident.get_start_time() // 10
            t_end = t_start + incident.get_resolution_time() // 10
            incident_list[t_start] = {'subsector_id': incident_subsector, 'start': t_start, 'end': t_end}

        return incident_list






