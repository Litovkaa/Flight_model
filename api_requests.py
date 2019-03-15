import requests
from credentials import *
from datetime import datetime
import time
from pandas.io.json import json_normalize
import pandas as pd

def request_scheduled_data(time_period):
    date_today = datetime.strptime(str(datetime.now()).split(" ")[0], "%Y-%m-%d")
    start_time = int(time.mktime(date_today.timetuple()))
    end_time = start_time + time_period
    payload = {'startDate': start_time,
               'endDate': end_time,
               'origin':"DME",
               'howMany': 15,
               'flightno':"DP414",
               'airline':"Pobeda"}
    response = requests.get(fxmlUrl + 'AirlineFlightSchedules',
                            params=payload, auth=(USERNAME, apiKey))

    if response.status_code == 200:
        return response.json()
    else:
        return "Error executing request"

def request