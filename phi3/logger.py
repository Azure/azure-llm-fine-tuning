import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set this to the lowest level you want to capture

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set this to the lowest level you want to capture

# Create file handler which logs even debug messages
file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)  # Set this to the lowest level you want to capture

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)