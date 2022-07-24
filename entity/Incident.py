class Incident(object):

    def __init__(self, start_time=None, resolution_time=None, location=None):

        self.start_time = start_time
        self.resolution_time = resolution_time
        self.location = location  # patrol area object

    def get_start_time(self):
        return self.start_time

    def get_resolution_time(self):
        return self.resolution_time

    def get_location(self):
        return self.location

    def get_sector(self):
        return self.location.get_sector()

    def to_string(self):
        return "<Incident Location=" + str(self.location.get_id()) + ">" + "\n" + \
               "<Incident Sector=" + str(self.get_sector()) + ">" + "\n" + \
               "<Incident Time=" + str(self.start_time) + ">" + "\n" + \
               "<Resolution Time=" + str(self.resolution_time) + ">"
