import logging
import os

class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger"""
    # If log file exists, truncate it to remove all contents
    if os.path.exists(log_file):
        open(log_file, 'w').close()
    
    logger = logging.getLogger(name)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logger.root.addHandler(console_handler)
    logger.root.setLevel(level)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.root.addHandler(file_handler)

    return logger
