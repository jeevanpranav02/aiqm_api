import datetime


# Create a class for logging the logging must include a timestamp format
class Logger:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def log(self, message: str):
        with open(self.log_file_path, "a") as log_file:
            log_file.write(f"{datetime.datetime.now()} : {message}\n")
