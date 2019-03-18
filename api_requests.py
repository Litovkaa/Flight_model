import requests
from credentials import *
from datetime import datetime
import time
from pandas.io.json import json_normalize
import pandas as pd

def request_scheduled_data(time_period, n_lines):
    date_today = datetime.strptime(str(datetime.now()).split(" ")[0], "%Y-%m-%d")
    start_time = int(time.mktime(date_today.timetuple()))
    end_time = start_time + time_period
    payload = {'startDate': start_time,
               'endDate': end_time,
               'origin':"DME",
               'howMany': n_lines}

    response = requests.get(fxmlUrl + 'AirlineFlightSchedules',
                            params=payload, auth=(USERNAME, apiKey))

    if response.status_code == 200:
        return response.json()
    else:
        return "Error executing request"

def request_aircraft_type(aircraft_type):
    payload = {'type':aircraft_type}

    response = requests.get(fxmlUrl + "AircraftType",
                            params=payload, auth=(USERNAME, apiKey))

    if response.status_code == 200:
        return response.json()
    else:
        return "Error executing request"

def request_airline_info(airline_code):
    payload = {"airlineCode":airline_code}

    response = requests.get(fxmlUrl + "AirlineInfo",
                            params=payload, auth=(USERNAME, apiKey))

    if response.status_code == 200:
        return response.json()
    else:
        return "Error executing request"

def compile_table(n_lines):
    if n_lines > 15:
        requests.get(fxmlUrl + 'SetMaximumResultSize',
                     params={'max_size':n_lines}, auth=(USERNAME, apiKey))

    scheduled_query = request_scheduled_data(86400, n_lines)
    schedule_df = json_normalize(scheduled_query['AirlineFlightSchedulesResult'], 'data')

    aircraft_types = {}
    for el in schedule_df.aircrafttype.unique():
        if el != "":
            aircraft_type = request_aircraft_type(el)["AircraftTypeResult"]
            aircraft_type = " ".join([aircraft_type['manufacturer'], aircraft_type['type']])
            aircraft_types[el] = aircraft_type

    airline_info_dict = {}
    airline_idents = [el[:3] for el in schedule_df.ident]
    for el in set(airline_idents):
        if el != "" and el not in airline_info_dict.keys():
            airline_info_dict[el] = request_airline_info(el)['AirlineInfoResult']['shortname']

    schedule_df.aircrafttype = [aircraft_types[el] if el != "" else None for el in schedule_df.aircrafttype.values]
    schedule_df['airline'] = [airline_info_dict[el] for el in airline_idents if el != ""]
    schedule_df.drop(columns=['meal_service', 'seats_cabin_business', 'seats_cabin_coach',
                              'seats_cabin_first', 'actual_ident'],
                     inplace=True)

    return schedule_df

