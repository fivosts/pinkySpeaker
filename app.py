#!/usr/bin/env python
"""
@package app 

This is the entry point of the application.
"""
from lib import dataloader as dl
from lib import model as m

import argparse as arg
import logging
import os

"""
Core function.
"""
def main():
	logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
	
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

	args = p.parse_args()

	if args.mode == "train":
		logging.info("Selected training of language model.")
		artist_list = [os.path.join(args.datapath, x) for x in args.train]
		dataset = dl.fetch_data(artist_list)
		model = m.simpleRNN(dataset)
	else:
		#TODO
		pass

	return

"""
Booting point of app
"""
if __name__ == "__main__":
	main()
	exit(0)