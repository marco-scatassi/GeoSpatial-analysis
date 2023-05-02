from datetime import datetime

# print INFO message with timestamp
def print_INFO_message_timestamp(message: str):
    now = datetime.now()
    timestamp = "[" + now.strftime("%m/%d/%y %H:%M:%S") + "]"
    print(f"{timestamp} INFO     {message}")

# print INFO message
def print_INFO_message(message: str):
    print(f"                    INFO     {message}")