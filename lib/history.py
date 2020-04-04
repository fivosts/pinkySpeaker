#!/usr/bin/env python

import sys, os, re
import collections
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.mrcrawley import AZspider as cr
from eupy.native import logger as l
from eupy.native import plotter as plt

## NN Training history object
## Keeps track of model type, model specs and loss, accuracy of trainig over epochs
class history:
    def __init__(self, ):
