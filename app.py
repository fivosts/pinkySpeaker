#!/usr/bin/env python
"""
@package app 

This is the entry point of the application.
"""
from lib import dataloader as dl

import argparse as arg
import logging
import os

"""
Checks the existence of requested dataset
"""
def check_dataset(path):
	for p in path:
		basename = os.path.basename(p)
		logging.info("Checking if {} dataset exists".format(basename))
		if not nos.isdir(p):
			logging.warning("{} dataset does not exist".format(basename))
			logging.info("Extracting dataset for {}".format(basename))
			dl.extract_artist(basename, p)
		else:
			logging.info("OK")
	return

"""
Checks whether a default generative model exists
"""
def check_model(path):

	return

def main():
	logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
	
	p = arg.ArgumentParser(description = "Song generator machine learning models")
	p.add_argument('-m', '--mode', default = "gen", choices = ["train", "gen"], required = False, help = "Choose between training a word model or generating songs")
	p.add_argument('-t', '--train', default = [], required = False, help = "Train model on selected artists")
	p.add_argument('-dp', '--datapath', default = "./dataset", required = False, help = "Path where datasets are stored")
	p.add_argument('-mp', '--modelpath', default = "./model", required = False, help = "Path where models are stored")

	args = p.parse_args()
	if args.mode == "train":
		check_dataset([os.path.join(args.datapath, x) for x in args.train])
		dl.
	else:
		check_model(args.modelpath)

	return

if __name__ == "__main__":
	main()
	exit(0)