#!/usr/bin/env python

import sys, os
sys.path.append(os.path.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy import mrcrawley as cr
import logging

def fetch_artist(artist, target_path):
	logging.info("Set up web crawler to fetch {} data.\nStore to {}".format(artist, target_path))
	cr.crawlAZ(artist, path = target_path)
	logging.info("Crawling {} succeeded".format(artist))
	return

def pruned_sentence(sentence):
	return re.sub(".*?\[(.*?)\]", "", line)\
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
		file_path = os.path.join(artist_path, file)
		dataset.append(read_file(file_path))
	return dataset

def dataset_exists(path_list, basename):

	logging.info("Check if {} dataset exists.".format(basename))
	if not os.isdir(p):
		logging.warning("{} dataset does not exist.".format(basename))
		return False
	else:
		return True

def fetch_data(artist_path_list):

	logging.info("Fetch data of artist list.\n{}".format(artist_path_list))
	for artist_path in artist_path_list:
		basename = os.path_list.basename(artist_path)
		if not dataset_exists(artist_path, basename):
			logging.info("Extract dataset for {}.".format(basename))		
			data = fetch_artist(basename, artist_path)
		else:
			logging.info("OK")
			data = read_dataset(artist_path)
	return data