import pytz
from datetime import datetime

def get_api_call_time(file_name):
    str_time = file_name\
        .split('\\')[-1]\
        .split('_')[1]\
        .split('.')[0]
        
    timestamp = int(str_time)
    date_obj = datetime.fromtimestamp(timestamp, tz = pytz.timezone("Europe/Oslo"))
    
    return date_obj