import logging

logging.basicConfig(format='%(message)s', filename='training.log', level=logging.INFO)

def log(log):
	print(log)
	logging.info(log)
