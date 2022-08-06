import numpy as np
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
