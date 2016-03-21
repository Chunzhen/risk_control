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
		self.path='C:/Users/chunzhen/Desktop/mbl/data/risk_control/'
		self.path_feature_type=self.path+'feature_type.csv'
		self.path_origin_train_x=self.path+'train/master.csv'
		#额外的文件
		self.path_origin_train_loginfo=self.path+'train/loginfo.csv'
		self.path_origin_train_updateinfo=self.path+'train/updateinfo.csv'

		self.path_origin_predict_x=self.path+'test/master.csv'
		self.path_origin_predict_loginfo=self.path+'test/loginfo.csv'
		self.path_origin_predict_updateinfo=self.path+'test/updateinfo.csv'
		#y值直接在x里卖弄
		self.path_origin_train_y=''
		self.path_origin_predict_y=''
		self.path_uid=''

		"""
		Analysis 输出的分析文件
		"""
		self.path_analysis=self.path+'analysis/'


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
		self.path_train=self.path+'output/train/'
		self.path_predict=self.path+'output/test/'

		