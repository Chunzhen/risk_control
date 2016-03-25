#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data

class Load_scale_data(object):
	def __init__(self,config):
		self.config=config


	def load_preprocessing(self,ftype,scale):
		"""
		读取预处理输出的特征文件
		"""
		X=pd.read_csv(self.config.path+ftype+'/master_'+scale+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		return np.array(X,dtype='int')


	def load_train_X(self):
		"""
		组合训练集多个特征文件
		"""
		scale='dumps'
		X=self.load_preprocessing('train', scale)
		scale='location'
		X2=self.load_preprocessing('train', scale)
		X=np.hstack((X,X2))
		print X.shape
		return X
	
	def load_train_X_separate(self):
		origin_instance=Load_origin_data(self.config)
		y=origin_instance.load_train_y()
		uid=origin_instance.load_train_uid()
		X=self.load_train_X()

		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]
		for i in range(len(y)):
			if y[i]==1:
				X_1.append(X[i])
				uid_1.append(uid[i])
			else:
				X_0.append(X[i])
				uid_0.append(uid[i])
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	def load_predict_X(self):
		"""
		组合测试集多个特征文件
		"""
		scale='dumps'
		X=self.load_preprocessing('test', scale)
		scale='location'
		X2=self.load_preprocessing('test', scale)
		X=np.hstack((X,X2))
		print X.shape
		return X
		pass
