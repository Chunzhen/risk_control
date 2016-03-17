#! /usr/bin/env python
#! -*- coding:utf-8 -*-

import os

"""
class Config
训练信息配置类
"""
class Config(object):
	def __init__(self):
		"""
		基本配置信息
		"""
		self.path=''
		self.path_origin_train_x=''
		self.path_origin_predict_x=''
		self.path_origin_train_y=''
		self.path_origin_predict_y=''
		self.path_uid=''

		"""
		Preprocessing 输出的特征文件
		"""

		"""
		fold random state
		"""
		self.fold_random_state=7
		self.n_folds=5

		"""
		输出
		"""
		self.path_train=''
		self.path_predict=''

		