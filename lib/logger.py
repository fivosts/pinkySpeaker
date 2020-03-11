#!/usr/bin/env python
import logging

def configLogger():
	global logger
	logger.setLevel(logging.INFO)

	# create console handler and set level to debug
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	# create formatter
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	# add formatter to ch
	ch.setFormatter(formatter)

	# add ch to logger
	logger.addHandler(ch)
	return

def shutdown():
	logging.shutdown()
	return

# create logger
logger = logging.getLogger('lyric_generator')
configLogger()
