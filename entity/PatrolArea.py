class PatrolArea(object):

    def __init__(self, Id=None, name=None, sub_sector=None, npc=None, sector=None, division=None, coordinates=None,
                 demands=None):

        self.id = Id
        self.name = name
        self.sub_sector = sub_sector
        self.npc = npc
        self.sector = sector
        self.division = division
        self.lon, self.lat = coordinates[0], coordinates[1]
        self.demands = demands

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_sub_sector(self):
        return self.sub_sector

    def get_sector(self):
        return self.sector

    def get_npc(self):
        return self.npc

    def get_division(self):
        return self.division

    def get_lat(self):
        return self.lat

    def get_lon(self):
        return self.lon

    def get_demands(self):
        return self.demands

    def to_string(self):
        return "<Patrol Area Id=" + str(self.id) + ">" + "\n" + \
               "<Patrol Area Name=" + self.name + ">"
