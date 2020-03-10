#!/usr/bin/env python

from eupy import mrcrawley as cr
import logging


def extract_artist(artist, target_path):
	cr.crawlAZ(artist, path = target_path)
	logging.info("Crawling {} succeeded".format(artist))
	return
