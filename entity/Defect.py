class Defect(object):

    def __init__(self, type=None, agents=None, location=None, interval=None, magnitude=None, sector=None):

        self.type = type
        self.agents = agents
        self.location = location
        self.interval = interval
        self.magnitude = magnitude
        self.sector = sector

    def get_type(self):
        return self.type

    def get_agents(self):
        return self.agents

    def get_location(self):
        return self.location

    def get_interval(self):
        return self.interval

    def get_value(self):
        return self.magnitude

    def get_sector(self):
        return self.sector

    def to_string(self):
        # Patrol consecutiveness defect
        if self.type == 1:
            return "<Defect Type=" + str(self.type) + ">" + "\n" + \
                   "<Defect Agents=" + str(self.agents) + ">" + "\n" + \
                   "<Defect Interval= [" + str(self.interval[0]) + "," + str(self.interval[1]) + "]>" + "\n" + \
                   "<Defect Magnitude= " + str(self.magnitude) + ">"
        # Minimum patrol presence defect
        if self.type == 2:
            return "<Defect Type=" + str(self.type) + ">" + "\n" + \
                   "<Defect Location=" + str(self.location) + ">" + "\n" + \
                   "<Defect Magnitude= " + str(self.magnitude) + ">"

