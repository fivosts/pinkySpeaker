#!/usr/bin/env python

import sys, os, re
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.mrcrawley import spider as cr

from eupy.native import logger as l

def fetch_artist(artist, target_path):
	l.getLogger().info("Set up web crawler to fetch {} data.".format(artist))
	l.getLogger().info("Store to {}.".format(target_path))
	cr.crawl(artist, path = target_path)
	l.getLogger().info("Crawling {} succeeded".format(artist))
	return

def pruned_sentence(sentence):
	return re.sub(".*?\[(.*?)\]", "", sentence)\
			.lower()\
			.replace("i'm", "i am").replace("it's", "it is")\
			.replace("isn't", "is not").replace("there's", "there is")\
			.replace("they've", "they have").replace("\n", " endline")\
			.replace("we've", "we have").replace("wasn't", "was not")\
			.replace(".", " . ").replace(",", " , ")\
			.replace("-", "").replace("\"", "")\
			.replace(":", "").replace("(", "")\
			.replace(")", "").replace("?", " ?")\
			.replace("!", " !")\
			.split()

def read_file(song_path):

	with open(song_path, 'r') as f:
		song = []
		for line in f:
			if line != "\n":
				sentence = pruned_sentence(line)
				if sentence:
					song.append(sentence)
		# Add endfile token to the end
		song[-1].append("endfile")

	# 0th and 1st lines of datapoint correspond to artist and title
	# The rest of it is the lyrics
	return {'artist': song[0], 'title': song[1], 'lyrics': song[2:]}

def read_dataset(artist_path):

	dataset = []
	for file in os.listdir(artist_path):
		file_path = pt.join(artist_path, file)
		dataset.append(read_file(file_path))
	return dataset

def dataset_exists(path, basename):

	l.getLogger().info("Check if {} dataset exists.".format(basename))
	if not pt.isdir(path):
		l.getLogger().warning("{} dataset does not exist.".format(basename))
		return False
	else:
		return True

def fetch_data(artist_path_list):

	l.getLogger().info("Fetch data of artist list.")
	for artist_path in artist_path_list:
		l.getLogger().info(artist_path)
		basename = pt.basename(artist_path)
		if not dataset_exists(artist_path, basename):
			l.getLogger().info("Extract dataset for {}.".format(basename))		
			data = fetch_artist(basename, artist_path)
		else:
			l.getLogger().info("OK")
			data = read_dataset(artist_path)
	return data