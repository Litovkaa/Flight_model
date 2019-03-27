import requests
from data.credentials import *
from datetime import datetime
import time
from pandas.io.json import json_normalize
import pandas as pd


def collect_airlines_data():
    icao_codes = requests.get(fxmlUrl + "AllAirlines", auth=(USERNAME, apiKey))

    if icao_codes.status_code == 200:
        icao_codes = icao_codes.json()
    else:
        return

    icao_codes = json_normalize(icao_codes['AllAirlinesResult'], 'data')
    icao_codes.dropna(inplace=True)
    icao_codes.rename(columns={0:"AIRLINE_ICAO"})
    icao_codes['shortnames'] = icao_codes.apply(lambda x: request_airline_info(x), axis=1)
    return icao_codes


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


def request_faFlightID(ident, departure_time):
    payload = {'ident': ident, "departureTime": departure_time}

    response = requests.get(fxmlUrl + "GetFlightID",
                            params=payload, auth=(USERNAME, apiKey))

    if response.status_code == 200:
        out_dict = response.json()
        if 'error' not in out_dict.keys():
            return out_dict['GetFlightIDResult']
        else:
            return None


def schedule_FlightEx_request():
    return


def flightsInTheAir(n_lines):
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

    schedule_df['faFlightID'] = schedule_df.apply(lambda x: request_faFlightID(x['ident'], x['departuretime']), axis=1)

    return schedule_df


def request_airport_arrived(icao, n_lines):
    payload = {'airport': icao, 'howMany': n_lines}

    response = requests.get(fxmlUrl + 'Arrived',
                            params=payload, auth=(USERNAME, apiKey))

    if response.status_code == 200:
        return response.json()
    else:
        return


def request_FlightInfoEx(faFlightID):
    payload = {'ident': faFlightID}

    response = requests.get(fxmlUrl + 'FlightInfoEx',
                            params=payload, auth=(USERNAME, apiKey))

    if response.status_code == 200:
        return response.json()
    else:
        return


def ArrivedFlights(icao, n_lines=15):
    if n_lines > 15:
        requests.get(fxmlUrl + 'SetMaximumResultSize',
                     params={'max_size':n_lines}, auth=(USERNAME, apiKey))

    arrived_query = request_airport_arrived(icao)
    arrived_df = json_normalize(arrived_query, ['ArrivedResult'], 'arrivals')
