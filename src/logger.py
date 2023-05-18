import logging
import os 
from datetime import datetime


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOG_DIR='Logs'
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH=os.path.join(os.getcwd(),LOG_DIR,LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH,
format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
level=logging.INFO
)

