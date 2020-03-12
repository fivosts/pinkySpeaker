#!/usr/bin/env python
import sys
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l

class simpleRNN:

	_logger = None

	def __init__(self, data = None):
		self._logger = l.getLogger()
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN.__init__()")
		self._initNNModel()
		#struct_sentences is only used for the word model
		# One function that will return title_set, lyric_set
		return

	def _initNNModel(self):
		self._logger.debug("pinkySpeaker.lib.model.simpleRNN._initNNModel()")
		self._logger.info("Initialize NN Model")
		## Any new sub-model should be registered here
		## The according function should be written
		self._model = { 'word_model'  : self._setWordModel(),
						'title_model' : self._setTitleModel(),
						'lyric_model' : self._setLyricModel() 
					  } 
		return 

	def _initWordModel(self):

		return

	def _initTitleModel(self):
		return

	def _initLyricModel(self):
		return
		