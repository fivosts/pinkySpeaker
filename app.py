#!/usr/bin/env python
"""
@package app 

This is the entry point of the application.
"""
from lib import dataloader as dl
from lib.models import simpleRNN, Transformer
import sys
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l
import argparse as arg
import os

MODEL_ZOO = {
				'simpleRNN'		: simpleRNN.simpleRNN,
				'Transformer'	: Transformer.Transformer
			}


"""
Argparse command line configuration
"""
def configArgs():

	p = arg.ArgumentParser(description = "Song generator machine learning models")
	p.add_argument('-md', '--model', 
					choices = ["simpleRNN", "Transformer"], required = True,
					help = "Choose model architecture for the sequence generation")
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
	p.add_argument('-ps', '--plot_samples', default = "", 
					required = False, choices = ["show", "save"],
					help = "Plot length of data samples")
	p.add_argument('-dbg', '--debug', default = False, 
					required = False, action = 'store_true',
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

	args = configArgs()
	logger = l.initLogger('lyric_generator', args.debug)
	logger.debug("pinkySpeaker.app.main()")
	setupFolders((args.datapath, args.modelpath))

	if args.mode == "train":
		logger.info("Selected training of language model.")
		artist_list = [os.path.join(args.datapath, x.lower()) for x in args.train]
		dataset = dl.fetchData(artist_list, args.plot_samples)
		model = MODEL_ZOO[args.model](data = dataset, LSTM_Depth = 8)
		loss = model.fit(epochs = 50)
	else: ## args.mode == "gen"
		model = MODEL_ZOO[args.model](model = args.modelpath)
		while(True):
			try:
				prediction_seed = input("Insert seed for sampling: ")
				model.predict(prediction_seed)
			except KeyboardInterrupt:
				logger.info("Terminating app...")

	logger.shutdown()
	return

"""
Booting point of app
"""
if __name__ == "__main__":
	main()
	exit(0)