#!/usr/bin/env python

import sys, os
from os import path as pt
sys.path.append(pt.dirname("/home/fivosts/PhD/Code/eupy/eupy"))
from eupy.native import logger as l

## NN Training history object
## Keeps track of model type, model specs and loss, accuracy of trainig over epochs
class history:
    def __init__(self, modeltype, **kwargs):
        self._logger = l.getLogger()
        self._logger.debug("pinkySpeaker.lib.history.history.__init__()")
        self._modeltype = modeltype
        self._kwargs = self._createProperty(kwargs)
        self._loss = []
        self._accuracy = []
        return

    @property
    def loss(self):
        self._logger.debug("pinkySpeaker.lib.history.history.loss_property()")
        return self._loss

    @property
    def accuracy(self):
        self._logger.debug("pinkySpeaker.lib.history.history.accuracy_property()")
        return self._accuracy
    
    @property
    def modeltype(self):
        self._logger.debug("pinkySpeaker.lib.history.history.modeltype()")
        return self._modeltype
    
    @loss.setter
    def loss(self, loss_element):
        self._logger.debug("pinkySpeaker.lib.history.history.loss_setter()")
        self._loss.append(loss_element)

    @accuracy.setter
    def accuracy(self, accuracy_element):
        self._logger.debug("pinkySpeaker.lib.history.history.accuracy_setter()")
        self._accuracy.append(accuracy_element)

    def _createProperty(self, props):
        self._logger.debug("pinkySpeaker.lib.history.history._createProperty()")
        print(props)
        return