import math
import numpy as np
import sys

from constants.Settings import RESPONSE_TIME, TIME_UNIT
from util.utils import reverse_mapping, round_to_nearest


# NOT IN USE
class SectorLite(object):

    def __init__(self, raw_sector=None):

        self.raw_sector = raw_sector

        # Build mapping table for the patrol area IDs
        self.map_to = {}
        self.map_from = {}
        self.table_by_sub_sector = {}
        self.table_by_patrol_area = {}
        self.sub_sectors = []
        self.all_petrol_areas = []
        self.presence_table = {}
        self.build_map_table()

        # Build simplified distance matrix
        size = len(self.map_to)
        self.distance_matrix = np.empty([size, size])
        self.build_distance_matrix()

    def build_map_table(self):

        i = 0
        self.map_to[self.raw_sector.get_hq().get_id()] = i

        all_patrol_areas = self.raw_sector.get_all_patrol_areas()

        for patrol_area in all_patrol_areas:
            i += 1
            self.map_to[patrol_area.get_id()] = i

        self.map_from = reverse_mapping(self.map_to)

        # For each sub sector, populate its corresponding a list of patrol area IDs
        for raw_sub_sectors in self.raw_sector.get_sub_sectors():
            self.sub_sectors.append(raw_sub_sectors.get_id())
            self.table_by_sub_sector[raw_sub_sectors.get_id()] = []
            for patrol_area in raw_sub_sectors.get_patrol_areas():
                self.table_by_sub_sector[raw_sub_sectors.get_id()].append(self.map_to[patrol_area.get_id()])

                # For each patrol area, compile a list of other patrol areas that is within a response time from itself
                self.table_by_patrol_area[self.map_to[patrol_area.get_id()]] = \
                    [self.map_to[x.get_id()] for x in all_patrol_areas if
                     self.raw_sector.get_distance_matrix()[patrol_area.get_id()][x.get_id()] < RESPONSE_TIME and
                     patrol_area.get_id() != x.get_id()]

                # For each patrol area, min presence
                self.presence_table[self.map_to[patrol_area.get_id()]] = \
                    raw_sub_sectors.get_master_table()[patrol_area.get_id()].get_demands()

    def build_distance_matrix(self):
        """
        Convert the raw distance matrix into a simplified np.array format and
        each distance is rounded up to its nearest 5
        :return: None
        """
        for i in self.raw_sector.get_distance_matrix().keys():
            for j in self.raw_sector.get_distance_matrix().keys():
                temp_distance = int(round_to_nearest(self.raw_sector.get_distance_matrix()[i][j], TIME_UNIT))
                self.distance_matrix[self.map_to[i], self.map_to[j]] = temp_distance

    def get_map_to(self):
        return self.map_to

    def get_map_from(self):
        return self.map_from

    def get_sub_sectors(self):
        return self.sub_sectors

    def get_all_patrol_areas(self):
        return list(self.get_table_by_petrol_area().keys())

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_table_by_petrol_area(self):
        return self.table_by_patrol_area

    def get_table_by_sub_sector(self):
        return self.table_by_sub_sector

    def get_presence_table(self):
        return self.presence_table

    def get_sector(self):
        return self.raw_sector
