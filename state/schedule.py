from copy import deepcopy

class Schedule(object):
    """

    """
    def __init__(self, T_max, num_agents, subsector_map, agents_map):

        self._T_max = T_max
        self._num_agents = num_agents
        self._subsector_map = subsector_map
        self._agents_map = agents_map
        self._state = {}
        self._state['initial_schedule'] = [[-1] * (self._T_max + 1) for _ in range(self._num_agents)]
        self._state['responded'] = [0]
        self._state['schedule'] = [[-1] * (self._T_max + 1) for _ in range(self._num_agents)]
        self._state['time_step'] = [0]
        self._state['incident_loc'] = [0]
        self._state['agent_travel_status'] = [0] * num_agents
        self._state['agent_travel_dest']  = [-1] * num_agents
        self._state['agent_arrival_time']  = [-1] * num_agents

    def update_state(self, **kwargs):

        for key, val in kwargs.items():
            if key in self._state.keys():
                self._state[key] = val
            else:
                raise KeyError

    def get_state(self, key):

        return deepcopy(self._state[key])

    def get_observation(self):

        return deepcopy((
            self._state['schedule'],
            self._state['time_step'],
            self._state['incident_loc'],
            self._state['responded'],
            self._state['agent_travel_status'],
            self._state['agent_travel_dest'],
            self._state['agent_arrival_time']
                ))

    def to_string(self):

        for agent_idx in range(self._num_agents):
            agent_id = self._agents_map['idx_2_agentid'][agent_idx]
            print ("================================================================")
            print (f"Timetable and status for Agent ID: {agent_id}, Index: {agent_idx}")
            print("================================================================")
            time_table = [self._subsector_map['idx_2_subsector'].get(subsector_idx, -1) for subsector_idx
                          in self._state['schedule'][agent_idx]]
            dest_idx = self._state['agent_travel_dest'][agent_idx]
            dest_id = self._subsector_map['idx_2_subsector'].get(dest_idx, -1)
            print (time_table)
            incident_loc_id = self._subsector_map['idx_2_subsector'].get(self._state['incident_loc'][0], -1)
            print (f"Incident id: {incident_loc_id}, timestep: {self._state['time_step'][0]}")
            print (f"Responded: {self._state['responded'][0]}")
            print (f"Travel Status: {self._state['agent_travel_status'][agent_idx]}")
            print(f"Travel Destination:{dest_id}, Index: {self._state['agent_travel_dest'][agent_idx]}")
            print(f"Travel Arrival Time: {self._state['agent_arrival_time'][agent_idx]}")
            print("")

    def get_reward(self):
        pass








