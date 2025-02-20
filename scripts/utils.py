import logging
import os

'''
Generates new log file for each client
'''
def log(msg, state):
    # create logger with 'spam_application'
    logger = logging.getLogger(state["clientId"])
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    if len(logger.handlers) == 0:
        filename = os.path.join(
            state["outputDirectory"], 'COINSTAC_%s.log' % state["clientId"])
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        logger.addHandler(fh)
    logger.info(msg)

