import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs", LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)


if __name__ == '__main__':
    logging.info("Logging has started")



# import logging
# import os
# from datetime import datetime

# # Define the directory for logs
# logs_dir = os.path.join(os.getcwd(), "logs")
# # Ensure the directory exists
# os.makedirs(logs_dir, exist_ok=True)

# # Define the log file name based on the current datetime
# LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
# # Full path for the log file
# LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )

# if __name__ == '__main__':
#     logging.info("Logging has started")
