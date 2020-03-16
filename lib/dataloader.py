#!/usr/bin/env python

import sys, os, re
import collections
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.mrcrawley import spider as cr
from eupy.native import logger as l
from eupy.native import plotter as plt

"""
Plot length of samples as bars.
Useful to determine the fixed size of sequences that will be fed to LSTM.
"""
def plotSamples(data, stream_out):
    l.getLogger().debug("pinkySpeaker.lib.dataloader.plotSamples()")

    stream_length = {}
    max_len = 0
    for d in data:
        ## dlen is the length of the song in num of tokens
        dlen = len(d['title']) + len([t for l in d['lyrics'] for t in l]) # Flattened lyrics list
        if dlen not in stream_length:
            stream_length[dlen] = 1
        else:
            stream_length[dlen] += 1
        max_len = max(max_len, dlen)

    ordered_list = collections.OrderedDict(sorted(stream_length.items())).items()
    num_chunks = 38
    chunk_size = int(max_len / (num_chunks - 1))

    plot_list = [{'x': [[x for x in range(chunk_size, num_chunks*chunk_size + chunk_size, chunk_size)]], 
                  'y': [[0] * num_chunks], 
                  'label': ["Song frequency per song length range"]}]
    
    for leng, freq in ordered_list:
        plot_list[0]['y'][0][int(leng / chunk_size)] += freq

    plt.plot_bars(plot_list,show_file = True, 
                            save_file = False,
                            bar_annotations = True,
                            show_xlabels = True)
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
def fetchData(artist_path_list, plot_sample):
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
    if plot_sample:
        plotSamples(data, plot_sample)
    print(plot_sample)

    return data