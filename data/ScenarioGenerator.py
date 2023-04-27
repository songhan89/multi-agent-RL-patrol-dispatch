import numpy as np
import pandas as pd
import pickle
import random
import sys

from scipy import stats

from constants.Settings import GAMMA_A, MIN_RESOLUTION_TIME, TIME_UNIT, SHIFT_DURATION
from entity.Incident import *
from util.utils import round_to_nearest


def generate_scenario(sector, n, poisson_mean):
    """

    :param sector: A sector object
    :param n: number of scenarios
    :param poisson_mean: the rate of occurrences of event per hour
    :return: A dictionary of scenarios with scenario index as key and list of incidents as values
    """

    scenarios = {}

    # Load the location probability distribution (pdf) of a given sector
    # pdf_table is a dictionary of {patrol area id: probability of incident occurences}
    with open("./data/location_pdf_" + sector.get_id() + ".pkl", "rb") as fp:
        pdf_table = pickle.load(fp)

    # Custom discrete pdf for patrol areas
    patrol_area_ids = list(pdf_table.keys())
    probabilities = list(pdf_table.values())
    custom_pdf = stats.rv_discrete(values=(patrol_area_ids, probabilities))

    # Iterate over each scenario
    for i in range(n):

        scenarios[i] = []
        # To indicate whether a scenario is successfully created
        done = False

        # Initialise the start time to be 0
        start_time = 0

        while not done:
            # Arrival time of incident modelled as poisson process
            # Generate inter-arrival time of incident to produce the incident start time
            inter_arrival_time = round_to_nearest(int(random.expovariate(poisson_mean / 60)), TIME_UNIT)
            start_time = start_time + inter_arrival_time

            # Resolution time modelled as gamma distribution
            resolution_time = stats.gamma.rvs(a=GAMMA_A, scale=TIME_UNIT, loc=MIN_RESOLUTION_TIME, size=1)
            resolution_time = round_to_nearest(resolution_time[0], TIME_UNIT)

            # Ignore the incident if it happens beyond the shift period
            if start_time + resolution_time > SHIFT_DURATION:
                done = True
                continue

            # Select the patrol area where incident occurs by following a pmf and add the incident instance into
            # the scenario
            location = sector.get_patrol_areas_table()[custom_pdf.rvs(size=1)[0]]
            scenarios[i].append(Incident(start_time, resolution_time, location))

    return scenarios


def compute_location_pdf(sector):
    """
    This function should only be called once when creating the initial sample schedule
    :param sector: A Sector Object
    :return: None, output a probability mass function of occurrences of dynamic event for each patrol area
    """

    patrol_areas = sector.get_all_patrol_areas()

    total_demands = 0
    pdf_table = {}

    # Use the number of road nodes in a patrol area as a proxy of demand in the area.
    # Add error factor to the demands of each area and compute the total demand
    for patrol_area in patrol_areas:
        pdf_table[patrol_area.get_id()] = patrol_area.get_demands() + int(1 + np.random.normal(0, 25) / 100)
        total_demands += pdf_table[patrol_area.get_id()]

    for patrol_area in patrol_areas:
        pdf_table[patrol_area.get_id()] = pdf_table[patrol_area.get_id()] / total_demands

    # Save the probability distribution function
    with open("./data/location_pdf_" + sector.get_id() + ".pkl", "wb") as fp:
        pickle.dump(pdf_table, fp)










