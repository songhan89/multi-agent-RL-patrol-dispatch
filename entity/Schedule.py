import numpy as np

from constants.Settings import T
from util.ScheduleUtil import *

class Schedule(object):

    def __init__(self, time_tables=None, time_intervals=None, objective_value=None, patrol_presence=None,
                 avail_indicators={}, patrol_indicators={}, unavailability_table={}):
        self.time_tables = time_tables
        self.time_intervals = time_intervals
        self.objective_value = objective_value
        self.patrol_presence = patrol_presence

        self.unavailability_table = unavailability_table

        self.avail_indicators = avail_indicators
        # If avail indicators are empty
        if not avail_indicators:
            for agent in time_tables:
                self.avail_indicators[agent] = [1] * len(T)

        self.patrol_indicators = patrol_indicators
        # if patrol indicators are empty
        if not patrol_indicators:
            for agent in time_tables:
                self.patrol_indicators[agent] = [1] * len(T)

    def get_time_tables(self):
        return self.time_tables

    def get_time_intervals(self):
        return self.time_intervals

    def get_agents(self):
        return self.time_tables.keys()

    def get_objective_value(self):
        return self.objective_value

    def get_unavailability_table(self):
        return self.unavailability_table

    def get_avail_indicators(self):
        return self.avail_indicators

    def update_time_tables(self, time_tables):
        """
        Update time table is inclusive of updating time intervals
        :param time_tables:
        :return:
        """
        self.time_tables = time_tables
        self.update_time_intervals(time_tables)

    def update_time_intervals(self, time_tables):
        time_intervals = {}

        for agent in time_tables.keys():
            # Convert the time table into a time interval format
            output = convert_to_time_intervals(time_tables[agent])
            time_intervals[agent] = output

        self.time_intervals = time_intervals

    def update_unavailability_table(self, agent, unavailability_schedule):
        """

        :param agent:
        :param unavailability_schedule: a tuple of location ID, start time index, end time index
        :return:
        """

        if agent not in self.unavailability_table:
            self.unavailability_table[agent] = []
        self.unavailability_table[agent].append(unavailability_schedule)

    # NOT IN USE for multi-agent problem
    def update_objective_value(self, new_obj_value):
        self.objective_value = new_obj_value

    def update_avail_indicators(self, agent, idx, value):
        self.avail_indicators[agent][idx] = value

    def convert_to_partial_schedule(self, t):

        for agent in self.time_tables.keys():
            self.time_tables[agent][0:t + 1] = [item for item in self.time_tables[agent][0:t + 1]]
            self.time_tables[agent][t + 1::] = [-1 for item in range(len(self.time_tables[agent][t + 1::]))]










