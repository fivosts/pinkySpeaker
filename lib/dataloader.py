#!/usr/bin/env python

from eupy import mrcrawley as cr

def gather_data():

	# Make a map of pink floyd html and dataset folder
	cr.crawlAZ("https://www.azlyrics.com/p/pinkfloyd.html", "./dataset/pink_floyd")
	return
