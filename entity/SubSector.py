# NOT IN USE
class SubSector(object):

    def __init__(self, Id=None, name=None, patrol_areas=None):
        self.id = Id
        self.name = name
        self.patrol_areas = patrol_areas

        self.master_table = {}
        self.build_master_table()

        self.adj_neighbours = []  # neighbouring subsectors

    def build_master_table(self):

        for patrol_area in self.patrol_areas:
            self.master_table[patrol_area.get_id()] = patrol_area

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_patrol_areas(self):
        return self.patrol_areas

    def get_patrol_areas_count(self):
        return len(self.patrol_areas)

    def get_master_table(self):
        return self.master_table

    def show_summary(self):
        return "<Sub Sector Id=" + str(self.id) + ">" + "\n" + \
               "<Sub Sector Name=" + str(self.name) + ">" + "\n" + \
               "<No. of Patrol Areas=" + str(self.get_patrol_areas_count()) + ">"
