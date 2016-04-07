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
		features=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		features_category=[]
		features_numeric=[]
		for i,t in enumerate(features['Index']):
			if features['Idx'][i]=='target':
				continue
			if t=='Categorical':
				features_category.append(features['Idx'][i])
			else:
				features_numeric.append(features['Idx'][i])
		return features_category,features_numeric

	def load_feature(self,tt):
		"""
		读取原始特征列的类型(numeric or category)
		"""
		features=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		r_features=[]
		for i,t in enumerate(features['Index']):
			if features['Idx'][i].lower().find(tt.lower())>=0 and t=='Categorical':
				r_features.append(features['Idx'][i])
		return r_features

	def load_train_X(self):
		"""
		读取训练集原始特征列
		"""
		features_category,features_numeric=self.load_feature_type()
		reader_category=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(features_category),encoding='utf-8')
		reader_numeric=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(features_numeric),encoding='utf-8')
		return reader_category,reader_numeric

	def load_predict_X(self):
		"""
		读取预测集原始特征列
		"""
		features_category,features_numeric=self.load_feature_type()
		reader_category=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(features_category),encoding='utf-8')
		reader_numeric=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(features_numeric),encoding='utf-8')
		return reader_category,reader_numeric

	def load_train_y(self):
		"""
		读取训练集的类标签
		"""
		y_reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(['target']),encoding='utf-8')
		y=np.array(y_reader,dtype='int')
		y=np.ravel(y)
		return y

	def load_train_X_separate(self):
		y=self.load_train_y()
		reader_category,reader_numeric=self.load_train_X()
		zero=[]
		one=[]
		for v in y:
			if v==0:
				zero.append(True)
				one.append(False)
			else:
				zero.append(False)
				one.append(True)

		reader_one_category=reader_category[one]
		reader_zero_category=reader_category[zero]
		reader_one_numeric=reader_numeric[one]
		reader_zero_numeric=reader_numeric[zero]
		return reader_one_category,reader_one_numeric,reader_zero_category,reader_zero_numeric

	def load_train_uid(self):
		"""
		读取训练集uid
		"""
		uid_reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(['Idx']),encoding='utf-8')
		uid=np.array(uid_reader,dtype='int')
		uid=np.ravel(uid)
		return uid

	def load_predict_uid(self):
		"""
		读取测试集的uid
		"""
		uid_reader=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(['Idx']),encoding='utf-8')
		uid=np.array(uid_reader,dtype='int')
		uid=np.ravel(uid)
		return uid



