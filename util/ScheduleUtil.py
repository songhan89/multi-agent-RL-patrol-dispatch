import sys

from copy import deepcopy

from scipy.spatial.distance import hamming

from constants.Settings import T, TIME_UNIT
from entity.Defect import *

from entity.Schedule import *
from util.utils import round_to_nearest, get_time_index, merge_dict, one_hot_encode, presence_utility_fn


def convert_to_time_intervals(time_table, k=0):
    """
    Convert a single time-table into a list of tuple [<patrol area id>, [start_time_idx, end_time_idx]]

    :param time_table:
    :param k:
    :return:
    """
    prev_elem = 1e7
    output = []
    temp_schedule = []

    time_table = time_table[k:]

    for idx in range(len(time_table)):

        if idx == 0:
            if time_table[idx] != prev_elem:
                temp_schedule.append(time_table[idx])
                temp_schedule.append([T[idx + k]])
        else:
            if time_table[idx] != prev_elem:
                #                 temp_schedule.append(input[idx])
                temp_schedule[1].append(T[idx + k])
                #                 print(temp_schedule)
                output.append(temp_schedule)
                temp_schedule = [time_table[idx], [T[idx + k]]]
                if idx == len(time_table) - 1:
                    temp_schedule[1].append(T[idx + k] + TIME_UNIT)
                    output.append(temp_schedule)
            else:
                if idx == len(time_table) - 1:
                    temp_schedule[1].append(T[idx + k] + TIME_UNIT)
                    output.append(temp_schedule)
                else:
                    prev_elem = time_table[idx]
                    continue

        prev_elem = time_table[idx]

    return output


def convert_to_util_intervals(array):
    """
    Convert an array of number of agents at every time step to
    a list of list [<no. of agents>, [start_time_idx, end_time_idx]]
    :param array:
    :return:
    """

    prev_elem = 1e7
    output = []
    temp = []

    for idx in range(len(array)):

        if idx == 0:
            if array[idx] != prev_elem:
                temp.append(array[idx])
                temp.append([idx])
        else:
            if array[idx] != prev_elem:
                temp[1].append(idx)
                output.append(temp)
                temp = [array[idx], [idx]]
            else:
                if idx == len(array) - 1:
                    temp[1].append(idx)
                    output.append(temp)
                else:
                    prev_elem = array[idx]
                    continue

        prev_elem = array[idx]

    return output


def check_time_table_consecutiveness(time_table, agent, time_matrix, k):
    """
    Check for type 1 defect in a single time-table.
    :param time_table:
    :param agent:
    :param time_matrix: use global time travel matrix if the problem is multi-agent
    :param k:
    :return:
    """
    time_interval = convert_to_time_intervals(time_table, k)

    # D = sector.get_distance_matrix()
    D = time_matrix
    defects = []

    for idx in range(len(time_interval) - 1):
        curr_loc = time_interval[idx][0]
        if curr_loc < 0:
            continue
        next_loc = time_interval[idx + 1][0]
        # Scenario where there is insufficient or excess time gap in between two patrol areas
        if next_loc < 0 and idx < len(time_interval) - 2:
            following_loc = time_interval[idx + 2][0]
            time_gap = time_interval[idx + 1][1][1] - time_interval[idx + 1][1][0]

            if get_time_index(round_to_nearest(D[curr_loc][following_loc] - time_gap, TIME_UNIT)) != 0:
                defects.append(
                    Defect(1, [agent], None, [get_time_index(time_interval[idx][1][1] - TIME_UNIT),
                                              get_time_index(time_interval[idx + 1][1][1])],
                           get_time_index(round_to_nearest(D[curr_loc][following_loc] - time_gap,
                                                           TIME_UNIT))))
        # Scenario where there is no travel time in between two patrol agents
        elif next_loc > 0:
            defects.append(
                Defect(1, [agent], None, [get_time_index(time_interval[idx + 1][1][0] - TIME_UNIT),
                                          get_time_index(time_interval[idx + 1][1][0])],
                       get_time_index(round_to_nearest(D[curr_loc][next_loc], TIME_UNIT))))

    return defects


def check_patrol_consecutiveness(schedule, time_matrix=None, k=0):
    """
    Check for type 1 defect in a schedule.
    :param schedule:
    :param time_matrix: use global time travel matrix if the problem is multi-agent
    :param k:
    :return:
    """
    defects = []

    for agent in schedule.get_time_tables():
        defects += check_time_table_consecutiveness(schedule.get_time_tables()[agent], agent, time_matrix, k)

    return defects


def get_old_jobs(schedule_dict, sector_id, t):
    old_jobs = set()
    time_table = schedule_dict[sector_id].get_time_tables()
    for agent in schedule_dict[sector_id].get_agents():
        patrol_area = time_table[agent][t]
        if patrol_area != -1:
            old_jobs.add(patrol_area)

    return old_jobs


def get_nearest_agent(schedule_dict, sector_id, incident, time_matrix, t):
    incident_loc = incident.get_location().get_id()
    time_table = schedule_dict[sector_id].get_time_tables()
    min_agent_dist = 1e7
    nearest_agent = None

    for agent in schedule_dict[sector_id].get_agents():
        agent_loc = time_table[agent][t-1]
        if agent_loc != -1:
            agent_dist = time_matrix[agent_loc][incident_loc]
            agent_dist = int(round_to_nearest(agent_dist, TIME_UNIT) / TIME_UNIT)
            if agent_dist < min_agent_dist:
                min_agent_dist = agent_dist
                nearest_agent = agent

    return nearest_agent, min_agent_dist


def get_effective_time_table(time_table, avail_indicator):
    """

    :param time_table:
    :param avail_indicator:
    :return: effective patrol time table where responding to an incident is indicated as 0 or not available
    """
    return (np.array(time_table) * np.array(avail_indicator)).tolist()


def get_effective_time_tables(schedule):
    """

    :param schedule:
    :return:
    """
    time_tables_eff = {}

    for agent in schedule.get_time_tables():
        time_tables_eff[agent] = get_effective_time_table(schedule.get_time_tables()[agent],
                                                          schedule.get_avail_indicators()[agent])

    return time_tables_eff


def get_patrol_count_table(time_tables):
    """

    :param time_tables:
    :return: a dictionary of {patrol area id : number of time period patrolled}
    """
    count_table = {}

    for agent in time_tables.keys():
        for item in time_tables[agent]:
            if item > 0:
                if item not in count_table:
                    count_table[item] = 0
                count_table[item] += 1

    return count_table


def check_min_patrol_presence(schedule, sector):
    """
    Check for type 2 defect. For single-agent problem only.
    :param schedule:
    :param sector:
    :return:
    """

    time_tables = schedule.get_time_tables()
    Q_l = sector.get_presence_table()
    # unavail_table = schedule.get_unavailability_table()

    time_tables_avail = get_effective_time_tables(schedule)

    count_table = get_patrol_count_table(time_tables_avail)
    # unavail_by_loc = {}
    defects = []

    # print(count_table)
    for key in Q_l.keys():
        min_patrol_presence = round_to_nearest(Q_l[key], TIME_UNIT) / TIME_UNIT
        if key not in count_table.keys():
            count_table[key] = 0
        if count_table[key] < min_patrol_presence:
            defects.append(Defect(2, location=key, magnitude=int(min_patrol_presence - count_table[key])))

    # sort defects by type. priority given to defect with highest magnitude
    defects = sorted(defects, key=lambda x: x.magnitude, reverse=True)

    return defects


def check_min_patrol_presence_MA(count_table, Q_j, patrol_areas_in_sector=[]):
    """
    Check for type 2 defect.
    :param count_table: in time period unit
    :param Q_j: in time period unit
    :return: a defect object
    """

    defects = []

    for key in Q_j.keys():
        # If need to check for defects by areas within a sector
        if patrol_areas_in_sector:
            if key not in patrol_areas_in_sector:
                continue
        if key not in count_table.keys():
            count_table[key] = 0
        if count_table[key] < Q_j[key]:
            defects.append(Defect(2, location=key, magnitude=int(Q_j[key] - count_table[key])))

    return defects


def check_defects(schedule, sector, k=0):
    """
    Given a schedule, check for defects in the schedule. For single-agent problem only.
    :param k: time index of incident response action
    :param schedule:
    :param sector:
    :return: list of defects
    """
    defects = []
    # check for patrol consecutiveness (type 1 defect)
    defects += check_patrol_consecutiveness(schedule, sector, k)

    # check for one agent per patrol area (type 2 defect)
    # defects += check_one_agent_per_area(schedule, sector)

    # check for min patrol presence (type 2 defect)
    defects += check_min_patrol_presence(schedule, sector)

    # schedule.update_defects(defects)
    return defects


def check_defects_MA(schedules_dict, sector_id, sectors, time_matrix, Q_j, k=0):
    """
    Given a dictionary of schedules, check for defects in the schedule. For multi-agent problem.
    :param schedules_dict:
    :param sector_id:
    :param sectors:
    :param Q_j:
    :param k:
    :return:
    """
    defects = []
    # check for patrol consecutiveness (type 1 defect)
    defects += check_patrol_consecutiveness(schedules_dict[sector_id], time_matrix, k)

    # check for min patrol presence (type 2 defect) only for the patrol areas within a sector
    count_table = get_patrol_count_table_MA(schedules_dict)
    patrol_areas = list(sectors[sector_id].get_presence_table().keys())
    defects += check_min_patrol_presence_MA(count_table, Q_j, patrol_areas)

    return defects


def compute_hamming_distance(schedule1, schedule2):
    """

    :param schedule1:
    :param schedule2:
    :return: total distance as a percentage (average)
    """
    time_tables1 = schedule1.get_time_tables()
    time_tables2 = schedule2.get_time_tables()

    total_distance = 0

    for agent in time_tables1:
        total_distance += hamming(time_tables1[agent], time_tables2[agent])

    return total_distance / len(time_tables1.keys())


def compute_max_hamming_distance(schedule1, schedule2):
    """

    :param schedule1:
    :param schedule2:
    :return: total distance as a percentage (max)
    """
    time_tables1 = schedule1.get_time_tables()
    time_tables2 = schedule2.get_time_tables()

    distances = []

    for agent in time_tables1:
        distances.append(hamming(time_tables1[agent], time_tables2[agent]))

    return max(distances)


def get_patrol_presence_status_by_area(schedule, sector, area_id):
    """
    Compute the ratio of patrol time in the area to the minimum requirement. For single-agent problem only.
    :param area_id:
    :param sector:
    :param schedule:
    :return: ratio
    """

    Q_j = sector.get_presence_table()

    time_tables_avail = get_effective_time_tables(schedule)

    count_table = get_patrol_count_table(time_tables_avail)

    min_patrol_presence = round_to_nearest(Q_j[area_id], TIME_UNIT) / TIME_UNIT

    if area_id not in count_table:
        return 0
    else:
        return count_table[area_id] / min_patrol_presence


def get_patrol_count_table_MA(schedules_dict):
    """

    :param schedules_dict:
    :param sectors:
    :return: A dictionary of how many time periods each patrol area has been patrolled
    """
    count_table = {}

    for sector_id in schedules_dict.keys():
        time_tables_avail = get_effective_time_tables(schedules_dict[sector_id])
        count_table = merge_dict(count_table, get_patrol_count_table(time_tables_avail))

    return count_table


def get_global_Q_j(sectors):
    """
    Generate a dictionary of {patrol area id: minimum number of patrol requirement in time period/index}
    :param sectors:
    :return: In terms of time period
    """
    Q_j = {}  # A dictionary with area id as key and the minimum patrol requirement as value

    for sector_id in sectors.keys():
        Q_j = merge_dict(Q_j, sectors[sector_id].get_presence_table())

    for key in Q_j:
        Q_j[key] = round_to_nearest(Q_j[key], TIME_UNIT) / TIME_UNIT

    return Q_j


def get_patrol_presence_status_by_area_MA(count_table, Q_j, area_id):
    """

    :param count_table:
    :param Q_j:
    :param area_id:
    :return: % of patrol presence over min patrol requirement of a patrol area
    """

    if area_id not in count_table:
        return 0
    else:
        return count_table[area_id] / Q_j[area_id]


def get_patrol_presence_status(schedule, sector):
    """
    For single-agent problem only.
    :param schedule:
    :param sector:
    :return: A list containing the % of patrol presence over min patrol requirement for all patrol areas
    in a given sector
    """

    patrol_status = []

    # Sort the patrol area id in ascending order
    sorted_area_ids = sorted(list(sector.get_patrol_areas_table().keys()))

    for area_id in sorted_area_ids:
        patrol_status.append(get_patrol_presence_status_by_area(schedule, sector, area_id))

    return patrol_status


def get_patrol_presence_status_MA(count_table, Q_j):
    """

    :param count_table:
    :param Q_j:
    :return: A list containing the % of patrol presence over min patrol requirement for all patrol areas
    across the sectors
    """
    patrol_status = []

    sorted_area_ids = sorted(list(Q_j.keys()))

    for area_id in sorted_area_ids:
        patrol_status.append(get_patrol_presence_status_by_area_MA(count_table, Q_j, area_id))

    return patrol_status


def get_patrol_presence_utility_by_area(area_id, schedule, sector):
    """
    For single-agent problem only.
    :param area_id:
    :param schedule:
    :param sector:
    :return:
    """

    Q_j = sector.get_presence_table()

    time_tables_avail = get_effective_time_tables(schedule)

    count_table = get_patrol_count_table(time_tables_avail)

    min_patrol_presence = round_to_nearest(Q_j[area_id], TIME_UNIT) / TIME_UNIT

    if area_id not in count_table:
        return 0
    else:
        return presence_utility_fn(count_table[area_id], min_patrol_presence)


def get_patrol_presence_utility(schedule, sector):
    """
    Compute the utility of patrol presence in a given sector. For single-agent problem only.
    :param schedule:
    :param sector:
    :return:
    """

    patrol_presence_util = 0

    # Sort the patrol area id in ascending order
    sorted_area_ids = sorted(list(sector.get_patrol_areas_table().keys()))

    for area_id in sorted_area_ids:
        patrol_presence_util += get_patrol_presence_utility_by_area(area_id, schedule, sector)

    return patrol_presence_util


def get_patrol_utilization_by_agent(time_table, avail_indicator):
    """

    :param time_table: patrol time table of a single agent
    :param avail_indicator:
    :return:
    """
    total_patrol_time = 0
    time_table_eff = get_effective_time_table(time_table, avail_indicator)

    for time_idx in range(len(T)):
        if time_table_eff[time_idx] > 0:
            total_patrol_time += 1

    return total_patrol_time / len(T)


def get_patrol_utilization(schedule):
    """
    Compute the proportion of time spent on patrolling in a shift
    :param schedule:
    :return: Proportion in terms of %
    """
    time_tables = schedule.get_time_tables()

    output = 0
    for agent in time_tables:
        output += get_patrol_utilization_by_agent(time_tables[agent], schedule.get_avail_indicators()[agent])

    return output / len(time_tables.keys())

    # return total_patrol_time / (len(T) * len(time_tables.keys()))


def get_objective_value(schedule, sector):
    """
    The fitness value of a schedule is defined as the proportion of effective patrol presence over total effective
    patrol time. For single-agent problem only.
    :param sector:
    :param schedule:
    :return:
    """

    total_effective_patrol = get_patrol_utilization(schedule) * (len(T) * len(schedule.get_time_tables().keys()))
    total_presence_utility = get_patrol_presence_utility(schedule, sector)

    return total_presence_utility / total_effective_patrol


def get_patrol_presence_utility_MA(schedules_dict, sectors):
    total_presence_utility = 0
    count_table = get_patrol_count_table_MA(schedules_dict)
    Q_j = get_global_Q_j(sectors)

    for area_id in count_table.keys():
        total_presence_utility += presence_utility_fn(count_table[area_id], Q_j[area_id])

    return total_presence_utility


def get_objective_value_MA(schedules_dict, sectors):
    total_effective_patrol = 0

    for sector_id in schedules_dict.keys():
        total_effective_patrol += get_patrol_utilization(schedules_dict[sector_id]) * \
                                  (len(T) * len(schedules_dict[sector_id].get_time_tables().keys()))

    total_presence_utility = get_patrol_presence_utility_MA(schedules_dict, sectors)

    print (total_presence_utility, total_effective_patrol)
    if total_effective_patrol == 0:
        return 0

    return total_presence_utility/total_effective_patrol


def get_post_state_MA(schedule, Q_j, all_patrol_areas, k, subagent_dim):
    """
    Post-Decision State consists of decision time index, patrol utilisation of each sectoral sub-agents,
    patrol statuses of all patrol areas, encoded schedules of each sectoral sub-agents
    :param schedule:
    :param Q_j:
    :param all_patrol_areas:
    :param k:
    :param subagent_dim: the dimension of the largest number of agent in the sectors.
    To ensure that the state size is the same for all sector/agent because every agent share the same network weights
    :return:
    """

    state = []

    # Time index of the current state
    # state.append(k / len(T))

    time_tables = schedule.get_time_tables()
    agent_count = len(time_tables.keys())

    # Number of dummy sub-agent to be created
    dummy_agent_count = int(subagent_dim - agent_count)

    # The patrol utilisation for each agent to represent the patrol time table
    for agent in schedule.get_time_tables():
        # state.append(schedule.get_time_tables()[agent][k])
        state.append(get_patrol_utilization_by_agent(time_tables[agent], schedule.get_avail_indicators()[agent]))

    # For each dummy agent, add the patrol utilization
    for add_agent in range(dummy_agent_count):
        state.append(0.0)

    time_tables_eff = get_effective_time_tables(schedule)

    # Add patrol status
    count_table = get_patrol_count_table(time_tables_eff)
    state += get_patrol_presence_status_MA(count_table, Q_j)

    # Add the encoded schedule to the state
    all_patrol_areas_copy = deepcopy(all_patrol_areas)
    all_patrol_areas_copy.insert(0, -1)  # Add a location dummy if agent is on the way
    all_patrol_areas_copy.insert(0, 0)  # Add a location dummy if agent is responding to case
    encoded_schedules = []
    for agent in schedule.get_time_tables():
        # Encode to one-hot vector
        encoded_schedules.append(get_one_hot_vector(time_tables_eff[agent], all_patrol_areas_copy))

    # For each dummy agent, add a dummy patrol schedule
    for add_agent in range(dummy_agent_count):
        dummy_time_table = [0] * len(T)  # 0 means the dummy agent is not available at a given time period
        encoded_schedules.append(get_one_hot_vector(dummy_time_table, all_patrol_areas_copy))

    encoded_schedules = np.array(encoded_schedules)

    state.append(encoded_schedules)

    return np.array(state)


def get_pre_state(schedule, Q_j, all_patrol_areas, incident):
    """
    Pre-Decision State consists of decision time index, patrol utilisation of each sectoral sub-agents,
    encoded locations of incident and the current location of each of the sectoral sub-agents,
    patrol statuses of all patrol areas, encoded schedules of each sectoral sub-agents
    :param schedule:
    :param k: time step
    :return:
    """
    state = []

    if incident:
        k = get_time_index(incident.get_start_time())
    else:
        # if no further incident, reach terminal state
        k = get_time_index(T[-1])

    # Time index of the current state
    state.append(k / len(T))

    time_tables = schedule.get_time_tables()

    # The patrol utilisation for each agent to represent the patrol time table
    for agent in schedule.get_time_tables():
        # state.append(schedule.get_time_tables()[agent][k])
        state.append(get_patrol_utilization_by_agent(time_tables[agent], schedule.get_avail_indicators()[agent]))

    time_tables_eff = get_effective_time_tables(schedule)

    # Add patrol status
    count_table = get_patrol_count_table(time_tables_eff)
    state += get_patrol_presence_status_MA(count_table, Q_j)
    all_patrol_areas.insert(0, -1)  # Add a location dummy if agent is on the way
    all_patrol_areas.insert(0, 0)  # Add a location dummy if agent is responding to case

    # Add encoded locations (incident location, locations of each agents
    encoded_locations = []

    # Add encoded incident location
    if incident:
        encoded_locations.append(
            one_hot_encode(all_patrol_areas.index(incident.get_location().get_id()), len(all_patrol_areas)))
    else:
        # Encode all zeros
        dummy_encoding = one_hot_encode(all_patrol_areas.index(-1), len(all_patrol_areas))
        dummy_encoding[0] = 0.0
        encoded_locations.append(dummy_encoding)

    # Add encoded agents' current locations
    for agent in schedule.get_time_tables():
        encoded_locations.append(one_hot_encode(all_patrol_areas.index(time_tables_eff[agent][k]),
                                                len(all_patrol_areas)))

    encoded_locations = np.array(encoded_locations)
    state.append(encoded_locations)

    # Add encoded schedules
    encoded_schedules = []
    for agent in schedule.get_time_tables():
        # Encode to one-hot vector
        encoded_schedules.append(get_one_hot_vector(time_tables_eff[agent], all_patrol_areas))

    encoded_schedules = np.array(encoded_schedules)

    state.append(encoded_schedules)

    return np.array(state)


def get_post_joint_state(schedules_dict, Q_j, all_patrol_areas, k, subagent_dim):
    joint_states = []
    for sector_id in schedules_dict.keys():
        local_state = get_post_state_MA(schedules_dict[sector_id], Q_j, all_patrol_areas, k, subagent_dim)
        joint_states.append(local_state)

    return np.array(joint_states)


# For Single Agent problem
def get_post_state(schedule, sector, k):
    """
    State consists of [ time_tables of each agent, patrol_util of each agent, patrol status of each patrol area,
    patrol penalty, list of all patrol areas + -1, no.of agents] -> The last
    :param schedule:
    :param k: time step
    :return:
    """

    state = []

    # Time index of the current state
    state.append(k / len(T))

    time_tables = schedule.get_time_tables()

    # The patrol utilisation for each agent to represent the patrol time table
    for agent in schedule.get_time_tables():
        # state.append(schedule.get_time_tables()[agent][k])
        state.append(get_patrol_utilization_by_agent(time_tables[agent], schedule.get_avail_indicators()[agent]))


    # Patrol penalty
    state.append(get_patrol_penalty(schedule, sector))

    # Patrol status of each patrol area
    state += get_patrol_presence_status(schedule, sector)

    # Add the encoded schedule to the state
    time_tables_eff = get_effective_time_tables(schedule)
    # Sort the patrol area id in ascending order
    sorted_area_ids = sorted(list(sector.get_patrol_areas_table().keys()))
    sorted_area_ids.insert(0, -1)  # Add a location dummy if agent is on the way
    sorted_area_ids.insert(0, 0)  # Add a location dummy if agent is responding to case

    encoded_schedules = []
    for agent in schedule.get_time_tables():
        # Encode to one-hot vector
        encoded_schedules.append(get_one_hot_vector(time_tables_eff[agent], sorted_area_ids))

    encoded_schedules = np.array(encoded_schedules)

    state.append(encoded_schedules)

    return np.array(state)


def get_one_hot_vector(time_table, unique_patrol_areas):
    x = []
    for item in time_table:
        x.append(one_hot_encode(unique_patrol_areas.index(item), len(unique_patrol_areas)))

    x = np.array(x)
    #     x = torch.from_numpy(x).float()
    return x


"""
NOT IN USE or Used in single-agent problem
"""


# def get_objective_value(schedule, sector):
#     """
#     The fitness value of a schedule is the sum of all effective patrol time period minus
#     the penalty cost in terms of number of time period short of the minimum requirement
#     :param sector:
#     :param schedule:
#     :return: Total effective patrol time minus penalty (in time period unit)
#     """
#
#     return (get_patrol_utilization(schedule) * (len(T) * len(schedule.get_time_tables().keys())) - \
#             get_patrol_penalty(schedule, sector)) / (len(T) * len(schedule.get_time_tables().keys()))

# def compute_patrol_presence(time_tables, unavailability_table={}):
#
#     total_patrol_time = 0
#
#     for agent in time_tables.keys():
#         for time_idx in range(len(T)):
#             if time_tables[agent][time_idx] > -1:
#                 total_patrol_time += 1
#
#     non_patrol_times = 0
#     if len(unavailability_table) > 0:
#         for agent in unavailability_table:
#             for item in unavailability_table[agent]:
#                 non_patrol_times += (item[2] - item[1])
#
#     return (total_patrol_time - non_patrol_times)/ (len(T) * len(time_tables.keys()))


def get_patrol_penalty(schedule, sector):
    """
    Compute the proportion of patrol time that falls short of the minimum requirement
    :param sector:
    :param schedule:
    :return: patrol time deficit
    """

    deficit = 0
    Q_l = sector.get_presence_table()

    time_tables_avail = get_effective_time_tables(schedule)

    count_table = get_patrol_count_table(time_tables_avail)

    for key in Q_l.keys():
        min_patrol_presence = round_to_nearest(Q_l[key], TIME_UNIT) / TIME_UNIT
        if key not in count_table.keys():
            count_table[key] = 0
        if count_table[key] < min_patrol_presence:
            deficit += (min_patrol_presence - count_table[key])

    return deficit


def aggregate_state(state):
    state_agg = None

    return state_agg
