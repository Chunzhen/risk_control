#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
import json

class Preprocessing(object):
	def __init__(self,config):
		self.config=config

	def load_data(self):
		"""
		特征处理
		"""
		instance=Load_origin_data(self.config)
		#读取feature type
		features_category,features_numeric=instance.load_feature_type()
		#读取train, predict数据
		reader_train_category,reader_train_numeric=instance.load_train_X()
		reader_predict_category,reader_predict_numeric=instance.load_predict_X()

		len_train=len(reader_train_category)
		len_predict=len(reader_predict_category)
		#合并数据
		reader_category=pd.concat([reader_train_category,reader_predict_category])
		reader_numeric=pd.concat([reader_train_numeric,reader_predict_numeric])

		i=0
		for feature in features_category:
			reader_category[feature]=reader_category[feature].apply(self._deal_nan)
			col=reader_category[feature].tolist()
			print feature
			col=set(col)
			tmp_dummys=pd.get_dummies(reader_category[feature])
			print tmp_dummys.shape
			i+=1
			# if i>10:
			# 	break
			

		pass
	def _deal_nan(self,n):
		n=str(n)
		n=n.strip()
		n=n.replace('市','')
		if n=='nan':
			return -1
		else:
			return n

	def _deal_nan2(self,n):
		n=str(n)
		n=n.strip()
		if n=='nan':
			pass
			#return ''
		else:
			return n

	def get_location(self):
		instance=Load_origin_data(self.config)
		#读取feature type
		features_category,features_numeric=instance.load_feature_type()
		#读取train, predict数据
		reader_train_category,reader_train_numeric=instance.load_train_X()
		reader_predict_category,reader_predict_numeric=instance.load_predict_X()

		len_train=len(reader_train_category)
		len_predict=len(reader_predict_category)
		#合并数据
		reader_category=pd.concat([reader_train_category,reader_predict_category])
		location_features=['UserInfo_2','UserInfo_7','UserInfo_4','UserInfo_8','UserInfo_24','UserInfo_20','UserInfo_19']
		for feature in location_features:
			reader_category[feature]=reader_category[feature].apply(self._deal_nan2)
			col=reader_category[feature].tolist()
			col=set(col)
			self.output_location(col,feature)

	def output_location(self,col,name):
		f=open(self.config.path_location+name+'.csv','wb')
		f.write(json.dumps(list(col)))
		# for location in col:
		# 	f.write(str(location)+'\n')
		f.close()

	def scale_func(self):
		"""
		处理函数
		"""
		pass

	def output_data(self):
		"""
		输出函数
		"""
		pass

	def run(self):
		"""
		实例函数
		"""
		pass

	



