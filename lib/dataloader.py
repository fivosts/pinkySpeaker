#!/usr/bin/env python

import sys, os, re
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.mrcrawley import spider as cr

from eupy.native import logger as l

"""
Send crawl request to gather artist data.
Returns specific artist dataset.
"""
def fetch_artist(artist):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.fetch_artist()")
	data = cr.crawl(artist)
	return data

"""
Reconstruct input sentence and return it.
"""
def pruned_sentence(sentence):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.pruned_sentence()")
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

"""
Reads a single song file and returns a dictionary with artist, title, lyric section.
"""
def read_file(song_path):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.read_file()")
	
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

"""
Iterates over all song files of a specific artist.
"""
def read_dataset(artist_path):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.read_dataset()")

	dataset = []
	for file in os.listdir(artist_path):
		file_path = pt.join(artist_path, file)
		dataset.append(read_file(file_path))
	return dataset

"""
Boolean check whether requested artist's dataset exists.
"""
def dataset_exists(path, basename):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.dataset_exists()")

	l.getLogger().info("Check if {} dataset exists.".format(basename))
	if not pt.isdir(path):
		l.getLogger().warning("{} dataset does not exist.".format(basename))
		return False
	else:
		return True

"""
Boot function.
Gets a requested artist and returns its data
"""
def fetch_data(artist_path_list):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.fetch_data()")

	l.getLogger().info("Fetch data of artist list.")
	for artist_path in artist_path_list:
		l.getLogger().info(artist_path)
		basename = pt.basename(artist_path)
		if not dataset_exists(artist_path, basename):
			l.getLogger().info("Extract dataset for {}.".format(basename))		
			data = fetch_artist(basename)
		else:
			l.getLogger().info("OK")
			data = read_dataset(artist_path)
	return data