#!/usr/bin/env python
"""
@package app 

This is the entry point of the application.
"""
from lib import dataloader as dl
from lib.models import simpleRNN, Transformer, TfTransformer

from eupy.native import logger as l
from eupy.native import plotter as plt

import argparse as arg
import os

MODEL_ZOO = {
                'simpleRNN'     : simpleRNN.simpleRNN,
                'Transformer'   : Transformer.Transformer,
                'TfTransformer' : TfTransformer.TfTransformer
            }


"""
Argparse command line configuration
"""
def configArgs():

    p = arg.ArgumentParser(description = "Song generator machine learning models")
    p.add_argument('-md', '--model', 
                    choices = [x for x in MODEL_ZOO], required = True,
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
        model1 = MODEL_ZOO[args.model](data = dataset, num_layers = 4, d_model = 128, dff = 128, num_heads = 4)
        model2 = MODEL_ZOO[args.model](data = dataset, num_layers = 2, d_model = 64, dff = 64, num_heads = 2)
        model3 = MODEL_ZOO[args.model](data = dataset, num_layers = 1, d_model = 32, dff = 32, num_heads = 1)
        model4 = MODEL_ZOO['simpleRNN'](data = dataset, LSTM_Depth = 8, sequence_length = 320)
        # loss = model.fit(save_model = args.modelpath)
        ylim = 0
        for loss1, loss2, loss3, loss4 in zip(model1.fit(), model2.fit(), model3.fit(), model4.fit(epochs = 200)):
            ylim = max(ylim, max(max(loss1), max(loss2), max(loss3), max(loss4)))
            plt.linesSingleAxis({model1.properties: {'y': loss1}, 
                                model2.properties: {'y': loss2}, 
                                model3.properties: {'y': loss3},
                                model4.properties: {'y': loss4},
                                }, 
                                y_label = ("Loss", 13), 
                                x_label = ("Epochs", 13), 
                                vert_grid = True,
                                plot_title = ("Loss over transformers", 18),
                                y_lim = ylim + 0.1*ylim, x_lim = 200, 
                                live = True)

    else: ## args.mode == "gen"
        model = MODEL_ZOO[args.model](model = args.modelpath)
        while(True):
            try:
                prediction_seed = input("Insert seed for sampling: ")
                model.predict(prediction_seed)
            except KeyboardInterrupt:
                logger.info("Terminating app...")

    logger.info("Application terminated successfully")
    logger.shutdown()
    return

"""
Booting point of app
"""
if __name__ == "__main__":
    main()
    exit(0)