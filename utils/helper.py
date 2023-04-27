import numpy as np
import math
from constants.Settings import *
from scipy.spatial.distance import hamming

def empty_timetable(time_table, t_incident):
    """
    Set sector at t >= timestep to be -1
    :param time_table:
    :param timestep:
    :return:
    """

    for agent_idx in range(len(time_table)):
        for t in range(len(time_table[0])):
            if t >= t_incident:
                time_table[agent_idx][t] = -1

    return time_table

def hamming_distance(schedule1, schedule2):
    """
    :param schedule1:
    :param schedule2:
    :return: total distance as a percentage (average)
    """

    total_dist = 0

    for agent_idx in range(len(schedule1)):
        total_dist += hamming(schedule1[agent_idx], schedule2[agent_idx])

    return total_dist / len(schedule1)

def response_utility_fn(tau_k):
    """
    Response utility reduces with lateness
    :param tau_k: response time in time unit
    :return:
    """
    return math.exp(-1 / BETA_R * max(0, tau_k - TAU_TARGET // 10))

def convert_Q_j_to_idx(Q_j, subsector_map):
    """
    Convert Q_j from subsector id to subsector index
    :param Q_j: Minimum patrol time for the subsectors
    :param subsector_map:
    :return:
    """

    Q_j_idx = {}
    for k, v in Q_j.items():
        subsector_idx, min_q = subsector_map['subsector_2_idx'][k], v
        Q_j_idx[subsector_idx] = min_q

    return Q_j_idx

def presence_utility_fn(real, required):
    """
    All patrol time within the patrol requirement has an utility of 1 while each additional patrol time unit beyond
    what is required has exponential decreasing value
    :param real: total patrol time (in time period)
    :param required: total patrol requirement (in time period)
    :return:
    """

    additional_time = int(max(0, real - required))

    if additional_time == 0:
        return real
    else:
        base_utility = required
        additional_utility = 0

        for additional_time_unit in range(1, additional_time + 1):
            additional_utility += additional_time_unit * math.exp(-1 / BETA_P * additional_time_unit)

        return base_utility + additional_utility
#
def get_objective_value_MA(time_table, sub_sectors_list, Q_j):
    total_patrol_time = 0

    total_patrol_time += (len(T) * len(time_table))
    total_presence_utility = get_patrol_presence_utility_MA(time_table, sub_sectors_list, Q_j)

    return total_presence_utility / total_patrol_time
#
def get_patrol_presence_utility_MA(time_table, sub_sectors_list, Q_j):
    total_presence_utility = 0
    count_table = {}

    for subsector_idx in range(len(sub_sectors_list)):
        count_table[subsector_idx] = sum([agent_schedule.count(subsector_idx) for agent_schedule in time_table])
        total_presence_utility += presence_utility_fn(count_table[subsector_idx], Q_j[subsector_idx])

    return total_presence_utility
