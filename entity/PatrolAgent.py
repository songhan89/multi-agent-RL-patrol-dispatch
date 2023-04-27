from util.utils import get_time_index

# NOT IN USE
class PatrolAgent(object):

    def __init__(self, Id = None, time_table=None):

        self.Id = Id
        self.time_table = time_table

    def get_id(self):
        return self.Id

    def get_time_table(self):
        return self.time_table

    # def get_schedule(self):
    #     return convert_to_schedule(self.time_table)

    def get_curr_location(self, sim_time):
        return self.time_table[get_time_index(sim_time)]

    def set_new_time_table(self, new_time_table):
        self.time_table = new_time_table