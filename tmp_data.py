#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os

from config import Config

class Tmp_data(object):
	"""
	用来预存训练中多线程产生的临时值
	"""
	def __init__(self,config):
		self.config=config

		self.predicts={}
		self.uids={}
		for i in range(self.config.n_folds):
			self.predicts[i]=[]
			self.uids[i]=[]

