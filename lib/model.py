#!/usr/bin/env python
import sys
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l

import numpy as np
import gensim

from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file

class simpleRNN:

    _logger = None

    def __init__(self, data = None):
        self._logger = l.getLogger()
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.__init__()")

        self._data = data
        self._model = None

        self._initNNModel(data)
        #struct_sentences is only used for the word model
        # One function that will return title_set, lyric_set
        return

    def _initNNModel(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initNNModel()")
        self._logger.info("Initialize NN Model")
        ## Any new sub-model should be registered here
        ## The according function should be written

        inp_sent, max_title_length, all_titles_length = self._constructSentences(raw_data)
        word_model = self._initWordModel(inp_sent)
        pretrained_weights = word_model.wv.vectors

        title_model = self._initTitleModel(pretrained_weights)
        lyric_model = self._initLyricModel(pretrained_weights)
        self._model = { 'word_model'  : word_model,
                        'title_model' : title_model,
                        'lyric_model' : lyric_model 
                      }
        self._logger.info("SimpleRNN Compiled successfully")


        title_set, lyric_set = self._constructTLSet(raw_data, max_title_length, all_titles_length)
        self._data = { 'word_model'     : inp_sent,
                       'title_model'    : title_set,
                       'lyric_model'    : lyric_set 
                     }

                     
        return 

    def _initWordModel(self, inp_sentences):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initWordModel()")
        wm = gensim.models.Word2Vec(inp_sentences, size = 300, min_count = 1, window = 4, iter = 200)
        self._logger.info("Word2Vec word model initialized")
        return wm

    def _initTitleModel(self, weights):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initTitleModel()")

        vocab_size, embedding_size = weights.shape
        tm = Sequential()
        tm.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[weights]))
        tm.add(LSTM(units=2*embedding_size, return_sequences=True))
        tm.add(LSTM(units=2*embedding_size))
        tm.add(Dense(units=vocab_size))
        tm.add(Activation('softmax'))
        tm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        self._logger.info("Title model initialized")
        return tm

    def _initLyricModel(self, weights):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initLyricModel()")

        vocab_size, embedding_size = weights.shape
        tm = Sequential()
        tm.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[weights]))
        tm.add(LSTM(units=2*embedding_size, return_sequences=True))
        tm.add(LSTM(units=2*embedding_size))
        tm.add(Dense(units=vocab_size))
        tm.add(Activation('softmax'))
        tm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        self._logger.info("Lyric model initialized")
        return tm

    def _listToChunksList(self, lst, n):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._listToChunksList()")
        chunk_list = []
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i: i + n])
        return chunk_list

    def _constructSentences(self, raw_data):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._constructSentences()")
        self._logger.info("Sentence preprocessing for word model")

        sentence_size = 10
        max_title_length = 0
        all_titles_length = 0
        words = []
        for song in raw_data:

            curr_title_length = len(song['title'])
            all_titles_length += curr_title_length - 1
            if curr_title_length > max_title_length:
                max_title_length = curr_title_length
            
            for word in song['title']:
                words.append(word)
            for sent in song['lyrics']:
                for word in sent:
                    words.append(word)
        return self._listToChunksList(words, sentence_size), max_title_length, all_titles_length

    def _constructTLSet(self, raw_data, max_title_length, all_titles_length):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN._constructTLSet()")

        title_set = {'input': np.zeros([all_titles_length, max_title_length], dtype=np.int32), 
                'output': np.zeros([all_titles_length], dtype=np.int32)}

        index = 0
        title_input = []
        title_expected_output = []
        lyric_input = []
        lyric_expected_output = []
        
        for song in raw_data:
            for curr_sent_size in range(len(song['title']) - 1):
                for current_index in range(curr_sent_size + 1):
                    title_set['input'][index][(max_title_length - 1) - curr_sent_size + current_index] = self.word2idx(song['title'][current_index])
                    title_set['output'][index] = self.word2idx(song['title'][current_index + 1])
                index += 1

            # flat_song = [song['title']] + song['lyrics']
            # flat_song = [" ".join(x) for x in [song['title']] + song['lyrics']]
            flat_song = " ".join([" ".join(x) for x in [song['title']] + song['lyrics']]).split()
            for indx in range(len(flat_song) - 4):
                lyric_input.append([self.word2idx(x) for x in flat_song[indx : indx + 4]])
                lyric_expected_output.append(self.word2idx(flat_song[indx + 4]))


        lyric_set = {'input': np.zeros([len(lyric_input), 4], dtype=np.int32), 
                     'output': np.zeros([len(lyric_input)], dtype=np.int32)}

        lyric_set['input'] = np.asarray(lyric_input, dtype = np.int32)
        lyric_set['output'] = np.asarray(lyric_expected_output, dtype = np.int32)

        return title_set, lyric_set

    def word2idx(self, word):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.word2idx()")
        return self._model['word_model'].wv.vocab[word].index
    def idx2word(self, idx):
        self._logger.debug("pinkySpeaker.lib.model.simpleRNN.idx2word()")
        return self._model['word_model'].wv.index2word[idx]
