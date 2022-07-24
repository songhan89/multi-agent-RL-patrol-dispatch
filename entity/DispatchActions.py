import numpy as np
import itertools
import random
from copy import deepcopy
from constants.Settings import NUM_DISPATCH_ACTIONS, DISPATCH_ACTIONS
from util.ScheduleUtil import *
from util.utils import response_utility_fn, round_to_nearest

class DispatchActions(object):

    def __init__(self, agents, time_matrix, all_patrol_areas, sector_id):

        """
        DISPATCH_ACTIONS = {
                'respond': 0,
                'continue': 1,
                'nearest_new_job': 2,
                'nearest_old_job': 3,
                'best_new_job': 4,
                'best_old_job': 5
        }
        """
        self.action_number = {}
        self.responded = False
        self.agents = agents
        self.n_agents = len(agents)
        self.time_matrix = time_matrix
        self.action_space = {}
        self.all_actions = set(DISPATCH_ACTIONS.keys())
        self.sector_id = sector_id
        self.travel_status = {}
        self.all_jobs = {patrol_area for patrol_area in all_patrol_areas}
        self.old_jobs = set()
        self.new_jobs = set()

        for agent in self.agents:
            self.travel_status[agent] = {'status': None, 'dest': None, 'arrival_time': None}
            self.action_space[agent] = set(DISPATCH_ACTIONS.keys())

        action_list = (self.all_actions  for agent in self.agents)
        idx = 0
        for joint_action in (list(itertools.product(*action_list))):
            self.action_number[joint_action] = idx
            idx += 1

        return

    def get_joint_action_list(self):
        action_list = (self.action_space[agent] for agent in self.agents)
        #keep respond to maximum of 1 agent
        feasible_list = []
        for feasible_joint_action in (list(itertools.product(*action_list))):
            if feasible_joint_action.count('respond') <= 1:
                feasible_list.append(feasible_joint_action)
        return feasible_list

    def get_joint_action_num(self, joint_action):
        return self.action_number[joint_action]

    def update_feasible_actions(self, incident=None):

        for agent in self.agents:
            if self.travel_status[agent]['status'] == 'patrol':
                self.action_space[agent] = self.all_actions
            elif self.travel_status[agent]['status'] == 'travel':
                self.action_space[agent] = set(['continue'])

            if incident is None:
                self.action_space[agent] = self.action_space[agent] - set(['respond'])

        return

    def update_travel_status(self, schedule_dict, sector_id, t):
        cur_time_table = schedule_dict[sector_id].get_time_tables()
        for agent in self.agents:
            if cur_time_table[agent][t] == -1:
                timestep = t + 1
                while timestep < len(T) and cur_time_table[agent][timestep] == -1:
                    timestep += 1

                if timestep == len(T):
                    self.travel_status[agent] = {'status': 'travel', 'dest': None,
                                                 'arrival_time': 1e7}
                else:
                    self.travel_status[agent] = {'status': 'travel', 'dest': cur_time_table[agent][timestep],
                                                     'arrival_time': timestep}

            else:
                # not travelling
                if cur_time_table[agent][t] == -1:
                    print ("??????????")
                    exit()
                self.travel_status[agent] = {'status': 'patrol', 'dest': cur_time_table[agent][t], 'arrival_time':None}

        return

    def get_travel_status(self):
        return self.travel_status

    def _update_jobs_status(self):
        self.old_jobs = set()
        for agent in self.agents:
            if self.travel_status[agent]['dest'] is not None:
                self.old_jobs.add(self.travel_status[agent]['dest'])
        self.new_jobs = self.all_jobs - self.old_jobs

    def act(self, action_name, prev_schedule_dict, cur_schedule_dict, sector_id, incident, t):

        response_utility = 0

        for agent, action in zip(self.agents, action_name):
            temp1 = deepcopy(cur_schedule_dict[sector_id].get_time_tables()[agent])
            temp2 = self.travel_status[agent]
            if action == 'respond':
                cur_schedule_dict, response_utility = self.respond(agent, cur_schedule_dict, sector_id, incident, t)
            if action == 'continue':
                cur_schedule_dict = self.cont_patrol(agent, cur_schedule_dict, sector_id, incident, t)
            if action == 'nearest_new_job':
                cur_schedule_dict = self.take_nearest_job('new', agent, cur_schedule_dict, sector_id, incident, t)
            if action == 'nearest_old_job':
                cur_schedule_dict = self.take_nearest_job('old', agent, cur_schedule_dict, sector_id, incident, t)
            if cur_schedule_dict[sector_id].get_time_tables()[agent][t] == -1 and self.travel_status[agent]['status'] == 'patrol':
                print ("----")
                print (action, agent, t, cur_schedule_dict[sector_id].get_time_tables()[agent], self.travel_status[agent])
                print ("what")
                print (action, agent, t, temp1, temp2)
                exit()

        return (cur_schedule_dict, response_utility, self)

    def _update_traveller_timetable(self, agent, time_table, t):

        if self.travel_status[agent]['arrival_time'] > t:
            time_table[agent][t] = -1
        elif self.travel_status[agent]['arrival_time'] <= t:
            time_table[agent][t] = self.travel_status[agent]['dest']
            self.travel_status[agent] = {'status': 'patrol', 'dest': time_table[agent][t], 'arrival_time': None}

        return time_table

    def cont_patrol(self, agent, cur_schedule_dict, sector_id, incident, t):
        time_table = cur_schedule_dict[sector_id].get_time_tables()
        travel_status = self.travel_status[agent]['status']

        if travel_status == 'travel':
            time_table = self._update_traveller_timetable(agent, time_table, t)

        if travel_status == 'patrol':
            time_table[agent][t] = self.travel_status[agent]['dest']

        # print ("Before")
        # print(time_table[agent])
        # cur_schedule_dict[sector_id].update_time_tables(time_table)
        # print("After")
        # print (time_table[agent])
        # print ("**************************************")

        return cur_schedule_dict

    def take_nearest_job(self, job_status, agent, cur_schedule_dict, sector_id, incident, t):

        job_list = []
        time_table = cur_schedule_dict[sector_id].get_time_tables()
        agent_loc = time_table[agent][max(t-1,0)]
        travel_status = self.travel_status[agent]['status']
        self._update_jobs_status()

        # if agent_loc == -1 and self.travel_status[agent]['status'] == 'patrol':
        #     time_table[agent][max(t - 1, 0)] = time_table[agent][t]
        #     agent_loc = time_table[agent][max(t - 1, 0)]
        #     if agent_loc == -1:
        #         print (time_table[agent])

        if job_status == 'new':
            job_list = list(self.new_jobs - {agent_loc})
        elif job_status == 'old':
            job_list = list(self.old_jobs - {agent_loc})

        if len(job_list) > 0:
            if travel_status == 'travel':
                time_table = self._update_traveller_timetable(agent, time_table, t)

            if travel_status == 'patrol':
                min_job_dist = 1e7
                nearest_job_loc = []
                for job_loc in job_list:
                    job_dist = self.time_matrix[agent_loc][job_loc]
                    job_dist = int(round_to_nearest(job_dist, TIME_UNIT) / TIME_UNIT)
                    if job_dist <= min_job_dist:
                        nearest_job_loc.append(job_loc)
                        min_job_dist = job_dist

                chosen_job_loc = random.choice(nearest_job_loc)

                if min_job_dist > 0:
                    self.travel_status[agent] = {'status': 'travel', 'dest': chosen_job_loc,
                                                 'arrival_time': t + min_job_dist}
                    time_table[agent][t] = -1
                else:
                    self.travel_status[agent] = {'status': 'patrol', 'dest': chosen_job_loc, 'arrival_time': None}
                    time_table[agent][t] = chosen_job_loc

            cur_schedule_dict[sector_id].update_time_tables(time_table)
            return cur_schedule_dict
        else:
            return self.cont_patrol(agent, cur_schedule_dict, sector_id, incident, t)

    # def take_best_job(self, job_status, agent, cur_schedule_dict, sector_id, incident, t):
    #
    #     job_list = None
    #     time_table = cur_schedule_dict[sector_id].get_time_tables()
    #     agent_loc = time_table[agent][max(t-1,0)]
    #     self._update_jobs_status()
    #
    #     if job_status == 'new':
    #         job_list = list(self.new_jobs).remove(agent_loc)
    #     elif job_status == 'old':
    #         job_list = list(self.old_jobs).remove(agent_loc)
    #
    #     if len(job_list) > 0:
    #         if self.travel_status[agent]['status'] == 'travel':
    #             time_table = self._update_traveller_timetable(agent, time_table, t)
    #
    #         if self.travel_status[agent]['status'] == 'patrol':
    #             min_job_val = 1e7
    #             min_job_dist = 1e7
    #             nearest_job_loc = []
    #             for job_loc in job_list:
    #                 # job_val = self._get_job_value()
    #                 job_dist = self.time_matrix[agent_loc][job_loc]
    #                 job_dist = int(round_to_nearest(job_dist, TIME_UNIT) / TIME_UNIT)
    #                 if job_dist <= min_job_val and job_loc != agent_loc:
    #                     nearest_job_loc.append(job_loc)
    #                     min_job_dist = job_dist
    #
    #             chosen_job_loc = random.choice(nearest_job_loc)
    #
    #             if min_job_dist > 0:
    #                 self.travel_status[agent] = {'status': 'travel', 'dest': chosen_job_loc,
    #                                              'arrival_time': t + min_job_dist}
    #                 time_table[agent][t] = -1
    #             else:
    #                 self.travel_status[agent] = {'status': 'patrol', 'dest': chosen_job_loc, 'arrival_time': t}
    #                 time_table[agent][t] = chosen_job_loc
    #
    #         cur_schedule_dict[sector_id].update_time_tables(time_table)
    #         return cur_schedule_dict
    #     else:
    #         return self.cont_patrol(agent, cur_schedule_dict, sector_id, incident, t)


    def respond(self, agent, cur_schedule_dict, sector_id, incident, t):
        time_table = cur_schedule_dict[sector_id].get_time_tables()
        travel_status = self.travel_status[agent]['status']
        incident_loc = incident.get_location().get_id()
        agent_loc = time_table[agent][max(t-1,0)]
        response_utility = 0

        # print (f"incident: {t}:{incident_loc}")
        # print (f"Before: {time_table}")


        if self.travel_status[agent]['status'] is None:
            print ("This should not happen !")
            print (incident.to_string(), t)
            exit()

        if travel_status == 'travel':
            response_time = 1e7
            response_utility = 0
            time_table = self._update_traveller_timetable(agent, time_table, t)

        if travel_status == 'patrol' or agent_loc == incident_loc:
            response_time = self.time_matrix[agent_loc][incident_loc]
            response_time = int(round_to_nearest(response_time, TIME_UNIT) / TIME_UNIT)
            response_utility = response_utility_fn(response_time * TIME_UNIT)

            if response_time > 0:
                self.travel_status[agent] = {'status': 'travel', 'dest': incident_loc, 'arrival_time': t + response_time}
                time_table[agent][t] = -1
            else:
                self.travel_status[agent] = {'status': 'patrol', 'dest': incident_loc, 'arrival_time': None}
                time_table[agent][t] = incident_loc

        # print (f"After: {time_table}\n, {agent}\n, {self.travel_status[agent]}")
        # exit()

        cur_schedule_dict[sector_id].update_time_tables(time_table)

        return cur_schedule_dict, response_utility


    # def act(self, action_name, prev_schedule_dict, cur_schedule_dict, sector_id, incident, t, time_matrix):
    #
    #     prev_schedule_dict = deepcopy(prev_schedule_dict)
    #     cur_schedule_dict = deepcopy(cur_schedule_dict)
    #     time_table = cur_schedule_dict[sector_id].get_time_tables()
    #
    #     if action_name == 'nearest_respond':
    #         agent_chosen, response_time = get_nearest_agent(cur_schedule_dict, sector_id, incident, time_matrix, t)
    #         if agent_chosen is None:
    #             return (cur_schedule_dict, 0, self)
    #
    #         response_utility = response_utility_fn(response_time * TIME_UNIT)
    #         if response_time > 0:
    #             self.travel_status[agent_chosen] = {'status': -1, 'dest': incident.get_location().get_id(),
    #                                                 'arrival_time': t + response_time}
    #             time_table[agent_chosen][t] = - 1
    #         else:
    #             self.travel_status[agent_chosen] = {'status': 1, 'dest': incident.get_location().get_id(),
    #                                                 'arrival_time': t + incident.get_resolution_time() // TIME_UNIT}
    #             time_table[agent_chosen][t] = incident.get_location().get_id()
    #
    #         cur_schedule_dict[sector_id].update_time_tables(time_table)
    #
    #         for agent in cur_schedule_dict[sector_id].get_agents():
    #             if agent != agent_chosen:
    #                 cur_schedule_dict = self.copy_initial(agent, prev_schedule_dict, cur_schedule_dict,
    #                                                                   sector_id, t)
    #
    #
    #         # print ("-------------------------------------------------------------")
    #         # print (prev_schedule_dict[sector_id].get_time_tables()[agent_chosen])
    #         # print(cur_schedule_dict[sector_id].get_time_tables()[agent_chosen], self.travel_status[agent_chosen],
    #         #       incident.get_location().get_id(), incident.get_start_time(), agent_chosen)
    #         # print("-------------------------------------------------------------")
    #         return (cur_schedule_dict, response_utility, self)
    #
    #     if action_name == 'nearest_new_job':
    #         self.old_jobs = get_old_jobs(cur_schedule_dict, sector_id, t-1)
    #         self.new_jobs = self.all_jobs - self.old_jobs
    #         for agent in cur_schedule_dict[sector_id].get_agents():
    #             cur_schedule_dict = self.take_nearest_new_job(agent, prev_schedule_dict,
    #                                                            cur_schedule_dict,
    #                                                            sector_id, time_matrix, t)
    #
    #     if action_name == 'continue':
    #         for agent in cur_schedule_dict[sector_id].get_agents():
    #             cur_schedule_dict = self.copy_prev_timestep(agent, prev_schedule_dict, cur_schedule_dict,
    #                                                                     sector_id, t)
    #
    #     return (cur_schedule_dict, 0, self)
    #
    # def take_nearest_new_job(self, agent, prev_schedule_dict, cur_schedule_dict, sector_id, time_matrix, t):
    #
    #     time_table = cur_schedule_dict[sector_id].get_time_tables()
    #     agent_loc = time_table[agent][t-1]
    #     max_dist = 1e7
    #     nearest_job = -1
    #     if len(self.new_jobs) > 0 and agent_loc != -1:
    #         for job_loc in self.new_jobs:
    #             agent_dist = time_matrix[agent_loc][job_loc]
    #             agent_dist = int(round_to_nearest(agent_dist, TIME_UNIT) / TIME_UNIT)
    #
    #             if agent_dist < max_dist:
    #                 max_dist = agent_dist
    #                 nearest_job = job_loc
    #
    #         self.old_jobs.add(nearest_job)
    #         self.new_jobs.remove(nearest_job)
    #
    #         self.travel_status[agent] = {'status': 1, 'dest': nearest_job,
    #                                             'arrival_time': t + agent_dist}
    #         time_table[agent][t] = -1
    #
    #         cur_schedule_dict[sector_id].update_time_tables(time_table)
    #         return cur_schedule_dict
    #     else:
    #         # if there are no new jobs or unable to take new job, simply return 'continue' strategy
    #         return self.copy_prev_timestep(agent, prev_schedule_dict, cur_schedule_dict, sector_id, t)
    #
    #
    # def copy_prev_timestep(self, agent, prev_schedule_dict, cur_schedule_dict, sector_id, t):
    #     prev_time_tables = prev_schedule_dict[sector_id].get_time_tables()
    #     cur_time_tables = cur_schedule_dict[sector_id].get_time_tables()
    #     if cur_time_tables[agent][t-1] != -1:
    #         cur_time_tables[agent][t] = cur_time_tables[agent][t-1]
    #     else:
    #         #if travelling with a destination in mind
    #         if self.travel_status[agent]['status'] == -1 and self.travel_status[agent]['dest'] != -1:
    #             # if reached destination by current time step
    #             if self.travel_status[agent]['arrival_time'] == t:
    #                 cur_time_tables[agent][t] = self.travel_status[agent]['dest']
    #             # continue travelling if otherwise
    #             else:
    #                 cur_time_tables[agent][t] = -1
    #         else:
    #             cur_time_tables[agent][t] = prev_time_tables[agent][t]
    #
    #     cur_schedule_dict[sector_id].update_time_tables(cur_time_tables)
    #
    #     return cur_schedule_dict
    #
    # def copy_initial(self, agent, prev_schedule_dict, cur_schedule_dict, sector_id, t):
    #     prev_time_tables = prev_schedule_dict[sector_id].get_time_tables()
    #     cur_time_tables = cur_schedule_dict[sector_id].get_time_tables()
    #     cur_time_tables[agent][t] = prev_time_tables[agent][t]
    #     cur_schedule_dict[sector_id].update_time_tables(cur_time_tables)
    #
    #     if cur_time_tables[agent][t] != -1:
    #         self.travel_status[agent] = {'status': 1, 'dest': cur_time_tables[agent][t], 'arrival_time': t}
    #     else:
    #         self.travel_status[agent] = {'status': -1, 'dest': -1, 'arrival_time':-1}
    #
    #     return cur_schedule_dict



# DISPATCH_ACTIONS = {
#         'nearest_respond': 0,
#         'continue': 1,
#         'nearest_new_job': 2,
#         'nearest_old_job': 3,
#         'best_new_job': 4,
#         'best_old_job': 5
# }