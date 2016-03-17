#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config

class Load_origin_data(object):
	def __init__(self,config):
		self.config=config

	def load_feature_type(self):
		"""
		读取原始特征列的类型(numeric or category)
		"""
		pass

	def load_train_X(self):
		"""
		读取训练集原始特征列
		"""
		pass

	def load_predict_X(self):
		"""
		读取预测集原始特征列
		"""
		pass

	def load_train_y(self):
		"""
		读取训练集的类标签
		"""
		pass

