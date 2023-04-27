from constants.Settings import TAU_TARGET, TIME_UNIT, T
from util.utils import round_to_nearest


class Sector(object):

    def __init__(self, Id=None, sub_sectors=None, all_patrol_areas=None, distance_matrix=None):
        self.id = Id
        self.sub_sectors = sub_sectors
        self.all_patrol_areas = all_patrol_areas
        # self.hq = hq  # patrol area object
        self.distance_matrix = distance_matrix

        self.patrol_areas_table = {}  # a dictionary with patrol area ID as key and the respective patrol area as value
        self.presence_table = {}  # a dictionary with patrol area ID as key and the respective demand as value
        self.neighbours_table = {}  # a dictionary with patrol area ID as key and the list of patrol areas in
        # the sector that are within TAU_TARGET from itself
        self.proximity_table = {}  # a dictionary with patrol area ID as key and another dictionary with no of
        # TIME_UNITs as key and list of patrol areas within that distance

        # Populate the relevant tables
        self.build_tables()

        self.agents = [agent.get_id() for agent in self.sub_sectors]  # list of agent ID or subsector ID

    def build_tables(self):

        # for sub_sector in self.sub_sectors:
        #     self.master_table[sub_sector.get_id()] = sub_sector

        for patrol_area in self.all_patrol_areas:
            self.patrol_areas_table[patrol_area.get_id()] = patrol_area
            self.presence_table[patrol_area.get_id()] = patrol_area.get_demands()
            # For each patrol area, compile a list of other patrol areas that is within a response time from itself
            self.neighbours_table[patrol_area.get_id()] = [neighbour for neighbour in self.all_patrol_areas
                                                           if self.distance_matrix
                                                           [patrol_area.get_id()][neighbour.get_id()] < TAU_TARGET
                                                           and patrol_area.get_id() != neighbour.get_id()]

            self.proximity_table[patrol_area.get_id()] = {}
            for i in range(1, len(T)):
                self.proximity_table[patrol_area.get_id()][i] = \
                    [neighbour for neighbour in self.all_patrol_areas
                     if round_to_nearest(self.distance_matrix[patrol_area.get_id()][neighbour.get_id()],
                                         TIME_UNIT) <= i * TIME_UNIT
                     and patrol_area.get_id() != neighbour.get_id()]

    def update_proximity_table(self, sectors, time_matrix):

        external_patrol_areas = []
        for sector_id in sectors.keys():
            if sector_id != self.get_id():
                external_patrol_areas += sectors[sector_id].get_all_patrol_areas()

        for patrol_area in external_patrol_areas:
            self.proximity_table[patrol_area.get_id()] = {}
            for i in range(1, len(T)):
                self.proximity_table[patrol_area.get_id()][i] = \
                    [neighbour for neighbour in external_patrol_areas
                     if
                     round_to_nearest(time_matrix[patrol_area.get_id()][neighbour.get_id()], TIME_UNIT) <= i * TIME_UNIT
                     and patrol_area.get_id() != neighbour.get_id()]

    def get_id(self):
        return self.id

    def get_sub_sectors(self):
        return self.sub_sectors

    def get_hq(self):
        return self.hq

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_sub_sectors_count(self):
        return len(self.sub_sectors)

    def get_all_patrol_areas(self):
        return self.all_patrol_areas

    def get_patrol_areas_count(self):
        return len(self.all_patrol_areas)

    def get_patrol_areas_table(self):
        return self.patrol_areas_table

    def get_presence_table(self):
        return self.presence_table

    def get_neighbours_table(self):
        return self.neighbours_table

    def get_proximity_table(self):
        return self.proximity_table

    def get_agents(self):
        return self.agents

    def get_agents_count(self):
        return len(self.agents)

    def add_agent(self, new_agent_id):
        self.agents.append(new_agent_id)

    def show_summary(self):
        return "<Sector Id=" + str(self.id) + ">" + "\n" + \
               "<No. of Sub Sectors=" + str(self.get_sub_sectors_count()) + ">" + "\n" + \
               "<No. of Patrol Areas=" + str(self.get_patrol_areas_count()) + ">"
