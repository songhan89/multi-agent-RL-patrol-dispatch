import sys


class Dataset(object):

    def __init__(self, sectors=None, time_matrix=None, adj_matrix=None):

        self.sectors = sectors  # A list of sector objects
        self.time_matrix = time_matrix
        self.adj_matrix = adj_matrix # A dataframe

        self.master_table = {}
        self.build_master_table()

    def build_master_table(self):
        """

        :return: a dictionary of {sector id: sector object}
        """

        for sector in self.sectors:
            self.master_table[sector.get_id()] = sector

    def get_all_patrol_areas(self):
        """

        :return: a list of patrol area objects
        """

        all_patrol_areas = []
        for sector in self.sectors:
            all_patrol_areas += sector.get_all_patrol_areas()

        return all_patrol_areas

    def get_sectors_count(self):
        return len(self.sectors)

    def get_sub_sectors_count(self):
        """

        :return: Number of sub-sectors / a proxy of number of patrol agents
        """
        sub_sectors_count = 0
        for sector in self.sectors:
            sub_sectors_count += sector.get_sub_sectors_count()

        return sub_sectors_count

    def get_patrol_areas_count(self):
        return len(self.get_all_patrol_areas())

    def get_sectors(self):
        return self.sectors

    def get_master_table(self):
        return self.master_table

    def get_time_matrix(self):
        return self.time_matrix

    def get_adj_matrix(self):
        return self.adj_matrix

    def show_summary(self, level=0):

        print("<No. of Sectors=" + str(self.get_sectors_count()) + ">" + "\n" +
              "<No. of Sub Sectors=" + str(self.get_sub_sectors_count()) + ">" + "\n" +
              "<No. of Patrol Areas=" + str(self.get_patrol_areas_count()) + ">")

        # if level == 0:
        #     return "<No. of Sectors=" + str(self.get_sectors_count()) + ">" + "\n" + \
        #            "<No. of Sub Sectors=" + str(self.get_sub_sectors_count()) + ">" + "\n" + \
        #            "<No. of Patrol Areas=" + str(self.get_patrol_areas_count()) + ">"
        if level > 0:
            print()
            for sector in self.sectors:
                print(sector.show_summary())
                print()
                if level > 1:
                    print()
                    for sub_sector in sector.get_sub_sectors():
                        print(sub_sector.show_summary())
