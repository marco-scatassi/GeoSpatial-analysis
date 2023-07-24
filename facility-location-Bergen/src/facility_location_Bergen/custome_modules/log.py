from datetime import datetime

# print INFO message with timestamp
def print_INFO_message_timestamp(message: str, log_file: str = None):
    now = datetime.now()
    timestamp = "[" + now.strftime("%m/%d/%y %H:%M:%S") + "]"
    output = f"{timestamp} INFO     {message}"
    if log_file == None:
        print(output)
    else:
        with open(log_file, "a") as f:
            f.write(output + "\n")


# print INFO message
def print_INFO_message(message: str, log_file: str = None):
    output = f"                    INFO     {message}"
    if log_file == None:
        print(output)
    else:
        with open(log_file, "a") as f:
            f.write(output + "\n")
