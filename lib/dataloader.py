#!/usr/bin/env python

import sys, os, re
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.mrcrawley import spider as cr
from eupy.native import logger as l
from eupy.native import plotter as plt

"""
Plot length of samples as bars.
Useful to determine the fixed size of sequences that will be fed to LSTM.
"""
def plotSamples(data):
	return

"""
Send crawl request to gather artist data.
Returns specific artist dataset.
"""
def fetchArtist(artist):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.fetchArtist()")
	data = cr.crawl(artist)
	return data

"""
Reconstruct input sentence and return it.
"""
def prunedSentences(sentence):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.prunedSentences()")
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
def readFile(song_path):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.readFile()")
	
	with open(song_path, 'r') as f:
		song = []
		for line in f:
			if line != "\n":
				sentence = prunedSentences(line)
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
def readDataset(artist_path):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.readDataset()")

	dataset = []
	for file in os.listdir(artist_path):
		file_path = pt.join(artist_path, file)
		dataset.append(readFile(file_path))
	return dataset

"""
Boolean check whether requested artist's dataset exists.
"""
def datasetExists(path, basename):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.datasetExists()")

	l.getLogger().info("Check if {} dataset exists.".format(basename))
	if not pt.isdir(path):
		l.getLogger().warning("{} dataset does not exist.".format(basename))
		return False
	else:
		return True

"""
Writes dataset to files
"""
def writeToFiles(data, artist_path):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.writeToFiles()")
	l.getLogger().info("Writing to path {}".format(artist_path))
	for d in data:
		with open(artist_path, 'w') as f:
			f.write("{}\n{}\n\n{}".format(d['artist'], 
										d['title'], 
										"\n".join(d['lyrics'])))
	return

"""
Boot function.
Gets a requested artist and returns its data.
"""
def fetchData(artist_path_list):
	l.getLogger().debug("pinkySpeaker.lib.dataloader.fetch_data()")

	l.getLogger().info("Fetch data of artist list.")
	for artist_path in artist_path_list:
		l.getLogger().info(artist_path)
		basename = pt.basename(artist_path)
		if not datasetExists(artist_path, basename):
			l.getLogger().info("Extract dataset for {}.".format(basename))		
			data = fetchArtist(basename)
			writeToFiles(data)
		else:
			l.getLogger().info("OK")
			data = readDataset(artist_path)
	return data