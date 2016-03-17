#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config

class Load_scale_data(object):
	def __init__(self,config):
		self.config=config


	def load_preprocessing(self,path):
		"""
		读取预处理输出的特征文件
		"""
		pass

	def load_train_X(self):
		"""
		组合训练集多个特征文件
		"""
		pass

	def load_predict_X(self):
		"""
		组合测试集多个特征文件
		"""
		pass
