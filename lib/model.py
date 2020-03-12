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
		return
