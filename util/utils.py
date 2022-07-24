import datetime
import haversine as hs
import json
import math
import numpy as np
import os
import sys
import requests

from collections import Counter

from constants.Settings import BETA_P, BETA_R, START_SHIFT, TIME_UNIT, T, TAU_TARGET


def calculate_haversine_distance(coord1, coord2):
    """
    Use to generate travel time matrix if real data is not available
    :param coord1: (lat, lon) in degrees
    :param coord2: (lat, lon) in degrees
    :return: haversine distance in km
    """
    return hs.haversine(coord1, coord2)


def query_osrm(_original, _destination):
    """
    Query the open street map data for travel time between two points.
    :param _original:
    :param _destination:
    :return:
    """
    url = 'http://router.project-osrm.org/route/v1/driving/'
    url += ';'.join([','.join(_original), ','.join(_destination)])
    response = requests.get(url)
    response_json = response.json()

    return response_json["routes"][0]["duration"] / 60


def calculate_real_travel_time(coord1, coord2):
    origin = [str(coord1[0]), str(coord1[1])]
    destination = [str(coord2[0]), str(coord2[1])]

    return query_osrm(origin, destination)


def to_real_time(sim_time):
    """
    Convert simulation time to real time
    :param sim_time:
    :return:
    """
    hours = int(sim_time / 60)
    mins = int(sim_time % 60)

    return str(START_SHIFT + hours) + ":" + (str(mins) if mins > 0 else "00")


def get_time_index(sim_time):
    return int(sim_time / TIME_UNIT)


def to_sim_time(time_index):
    return int(time_index * TIME_UNIT)


def round_to_nearest(n, m):
    return n + (m - n) % m


def response_utility_fn(tau_k):
    """
    Response utility reduces with lateness
    :param tau_k: response time in minutes
    :return:
    """
    return math.exp(-1 / BETA_R * max(0, tau_k - TAU_TARGET))


def presence_utility_fn(real, required):
    """
    All patrol time within the patrol requirement has an utility of 1 while each additional patrol time unit beyond
    what is required has exponential decreasing value
    :param real: total patrol time (in time period)
    :param required: total patrol requirement (in time period)
    :return:
    """

    additional_time = int(max(0, real - required))

    if additional_time == 0:
        return real
    else:
        base_utility = required
        additional_utility = 0

        for additional_time_unit in range(1, additional_time + 1):
            additional_utility += additional_time_unit * math.exp(-1 / BETA_P * additional_time_unit)

        return base_utility + additional_utility


def merge_dict(dict1, dict2):

    dict1_orig = Counter(dict1)
    dict2_orig = Counter(dict2)

    return dict1_orig + dict2_orig


def extract_matrix(df_matrix, name_list):
    """

    :param df_matrix: dataframe
    :param name_list: subset of the sector ids
    :return: a tuple of 2D np array and a dictionary to map the sector id to the column index
    """
    column_names = list(df_matrix.columns)
    mapping_table = {}

    for column_name in column_names:
        mapping_table[column_name] = column_names.index(column_name)

    np_matrix = df_matrix.to_numpy()
    new_len = len(name_list)
    new_np_matrix = np.empty(shape=(new_len, new_len), dtype='int')
    new_mapping_table = {}

    i = 0
    for name in name_list:
        idx = mapping_table[name]
        if name not in new_mapping_table.keys():
            new_mapping_table[name] = i
        new_idx = new_mapping_table[name]
        for other_name in name_list:
            if name != other_name:
                if other_name not in new_mapping_table.keys():
                    i += 1
                    new_mapping_table[other_name] = i
                old_other_idx = mapping_table[other_name]
                new_other_idx = new_mapping_table[other_name]
                new_np_matrix[new_idx][new_other_idx] = np_matrix[idx][old_other_idx]
            else:
                new_np_matrix[new_idx][new_idx] = 0

    return new_np_matrix, new_mapping_table


def one_hot_encode(x, n_classes):
    return np.eye(n_classes)[x]


# NOT IN USE
def export_problem(sector, scenarios):
    with open(os.path.join(sys.path[0], 'problem.txt'), 'w') as f:
        f.write("Sector Info")
        f.write(sector.get_sector().show_summary())
        f.write("\n\n")
        f.write("ID mapping")
        f.write(str(sector.get_map_to()))
        f.write("\n\n")
        f.write("Patrol areas assigned to each agent")
        f.write(str(sector.get_table_by_sub_sector()))
        f.write("\n\n")
        f.write("Neighbouring information")
        f.write(str(sector.get_table_by_petrol_area()))
        f.write("\n\n")
        f.write("Scenarios")
        f.write(str(scenarios))
        f.write("\n\n")
        for sub_sector_id in sector.get_sector().get_master_table().keys():
            sub_sector = sector.get_sector().get_master_table()[sub_sector_id]
            f.write(str(sub_sector_id) + "_" + sub_sector.get_name() + "\n")
            patrol_area_list = []
            for patrol_area in sub_sector.get_patrol_areas():
                patrol_area_list.append((patrol_area.get_id(), patrol_area.get_name()))
            f.write(str(patrol_area_list) + "\n")


def reverse_mapping(table):
    reversed_table = {}

    for key in table.keys():
        reversed_table[table[key]] = key

    return reversed_table


def mapped_back(input, map_table):
    output = []

    # Add one more key to represent unavailability / travelling
    map_table[-1] = -1

    for i in input:
        output.append(map_table[int(i)])

    return output

#
# def calculate_real_travel_time(coord1, coord2):
#     # URL to the API
#     baseURL = "https://developers.onemap.sg/privateapi/routingsvc/route?"
#     startURL = "start="
#     endURL = "&end="
#     suffixURL = "&routeType=drive&token="
#
#     origin_coor = str(coord1[0]) + "," + str(coord1[1])
#     dest_coor = str(coord2[0]) + "," + str(coord2[1])
#
#     # give a name to .json file
#     fo1 = 'temp.json'
#
#     expiry_time = float(1617084143)
#     token = TOKEN
#
#     d = datetime.datetime.now()
#     unixtime = float(time.mktime(d.timetuple()))
#
#     if unixtime >= expiry_time:
#         token, expiry_time = get_onemap_token()
#
#     URL = '%s%s%s%s%s%s%s' % (baseURL, startURL, origin_coor, endURL, dest_coor, suffixURL, token)
#     # query the api
#     try:
#         urllib.request.urlretrieve(URL, fo1)
#         # open the output .json file and retrieve the planning area
#         with open(fo1) as data_file:
#             data = json.load(data_file)
#             # save the planning area and the name of the location into a temporary dataframe
#             travel_time = data["route_summary"]["total_time"] / 60
#             data_file.close()
#             os.remove(fo1)
#     except:
#         travel_time = 1e7
#
#     return travel_time

# def get_onemap_token():
#
#     url = "https://developers.onemap.sg/privateapi/auth/post/getToken"
#
#     payload = "{\"email\":\"waldy.joe.2018@phdcs.smu.edu.sg\",\"password\":\"Galaxy_s8\"}"
#     headers = {
#         'Content-Type': 'application/json',
#         'Cookie': 'Domain=developers.onemap.sg; onemap2=CgAACmBhSdaD+AXGBTcNAg==; _toffuid=rB8E8GBhSdYl0jCHAyM0Ag=='
#     }
#
#     response = requests.request("POST", url, headers=headers, data=payload)
#
#     return str(response.json()['access_token']), float(response.json()['expiry_timestamp'])

