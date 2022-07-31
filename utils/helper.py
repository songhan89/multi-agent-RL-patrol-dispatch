def get_distance():
    pass

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


