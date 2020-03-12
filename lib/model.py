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
		self._initNNModel()
		#struct_sentences is only used for the word model
		# One function that will return title_set, lyric_set
		return

	def _initNNModel(self):
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initNNModel()")
		self._logger.info("Initialize NN Model")
		## Any new sub-model should be registered here
		## The according function should be written
		word_model = self._initWordModel()
		pretrained_weights = word_model.wv.vectors

		title_model = self._initTitleModel(pretrained_weights)
		lyric_model = self._initLyricModel(pretrained_weights)
		self._model = { 'word_model'  : word_model,
						'title_model' : title_model,
						'lyric_model' : lyric_model 
					  } 
		return 

	def _initWordModel(self):
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initWordModel()")
		inp_sent = self._constructSentences()
		wm = gensim.models.Word2Vec(inp_sent, size = 300, min_count = 1, window = 4, iter = 200)
		return wm

	def _initTitleModel(self, weights):
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initTitleModel()")

		vocab_size, embedding_size = weights.shape
		tm = Sequential(Embedding(input_dim=vocab_size, 
								  output_dim=embedding_size, 
								  weights=[weights]))
		tm.add(LSTM(units=2*embedding_size, return_sequences=True))
		tm.add(LSTM(units=2*embedding_size))
		tm.add(Dense(units=vocab_size))
		tm.add(Activation('softmax'))
		tm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
		return tm

	def _initLyricModel(self, weights):
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initLyricModel()")

		vocab_size, embedding_size = weights.shape
		tm = Sequential(Embedding(input_dim=vocab_size, 
								  output_dim=embedding_size, 
								  weights=[weights]))
		tm.add(LSTM(units=2*embedding_size, return_sequences=True))
		tm.add(LSTM(units=2*embedding_size))
		tm.add(Dense(units=vocab_size))
		tm.add(Activation('softmax'))
		tm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
		return tm

	def _listToChunksList(self, lst, n):
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN._listToChunksList()")
		chunk_list = []
		for i in range(0, len(lst), n):
			chunk_list.append(lst[i: i + n])
		return chunk_list

	def _constructSentences(self):
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN._constructSentences()")

		sentence_size = 10
		words = []
		for song in self._data:
			for word in song['title']:
				words.append(word)
			for sent in song['lyrics']:
				for word in sent:
					words.append(word)
		return self._listToChunksList(words, sentence_size)
