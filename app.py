#!/usr/bin/env python
"""
@package app 

This is the entry point of the application.
"""
from lib import dataloader as dl
from lib import model as m
import sys
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l

import argparse as arg
import os

"""
Argparse command line configuration
"""
def configArgs():

	p = arg.ArgumentParser(description = "Song generator machine learning models")
	p.add_argument('-m', '--mode', default = "gen", 
					choices = ["train", "gen"], required = False, 
					help = "Choose between training a word model or generating songs")
	p.add_argument('-t', '--train', default = [], 
					required = False, nargs = "?", action = "append",
					help = "Train model on selected artists")
	p.add_argument('-dp', '--datapath', default = "./dataset", 
					required = False, 
					help = "Base path of datasets")
	p.add_argument('-mp', '--modelpath', default = "./model", 
					required = False, 
					help = "Base path of models")
	p.add_argument('-dbg', '--debug', default = False, 
					required = False, 
					help = "Enable DEBUG information")
	return p.parse_args()

def setupFolders(folds):
	for f in folds:
		l.getLogger().info("Setting up {}".format(f))
		os.makedirs(f, exist_ok = True)
	return

"""
Core function.
"""
def main():

	logger = l.initLogger('lyric_generator')
	args = configArgs()
	setupFolders((args.datapath, args.modelpath))

	if args.mode == "train":
		logger.info("Selected training of language model.")
		artist_list = [os.path.join(args.datapath, x.lower()) for x in args.train]
		dataset = dl.fetch_data(artist_list)
		model = m.simpleRNN(dataset)
	else:
		#TODO
		pass

	logger.shutdown()
	return

"""
Booting point of app
"""
if __name__ == "__main__":
	main()
	exit(0)