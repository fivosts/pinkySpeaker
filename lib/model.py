#!/usr/bin/env python
import sys
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l

class simpleRNN:

	def __init__(self, data):
		l.getLogger().critical("This is a dummy class to suppress errors")
		return
