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

		#struct_sentences is only used for the word model
		# One function that will return title_set, lyric_set
		return
