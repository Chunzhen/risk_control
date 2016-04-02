#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
from loginfo import Loginfo

class Load_scale_data(object):
	def __init__(self,config):
		self.config=config


	def load_preprocessing(self,ftype,scale):
		"""
		读取预处理输出的特征文件
		"""
		X=pd.read_csv(self.config.path+ftype+'/master_'+scale+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		return np.array(X,dtype=float)


	def load_train_X(self):
		"""
		组合训练集多个特征文件
		"""
		dumps=['dumps','dumps_no_location','numeric12_add_median']
		infos=['loginfo1','loginfo_limit1','loginfo_limit3','loginfo_limit7','loginfo3']
		updates=['updateinfo1','updateinfo_limit1','updateinfo_limit3','updateinfo_limit7','updateinfo_time']
		others=['x','lr','category_num2','city_rank','coor','missing_scale']


		scale='lr'
		X2=self.load_preprocessing('train', scale)
		X=X2[:,:len(X2[0])-1]
		#X=np.hstack((X,X2))
		print X.shape

		scale='WeblogInfo_weight'
		X2=self.load_preprocessing('train', scale)
		#X=X2
		X=np.hstack((X,X2))
		print X.shape

		# scale='UserInfo_weight'
		# X2=self.load_preprocessing('train', scale)
		# #X=X2
		# X=np.hstack((X,X2))
		# print X.shape

		# scale='coor'
		# X2=self.load_preprocessing('train', scale)
		# #X=X2
		# X=np.hstack((X,X2))
		# print X.shape

		# scale='coor'
		# X2=self.load_preprocessing('train', scale)
		# X=np.hstack((X,X2))
		# print X.shape

		# for info in updates:
		# 	X2=self.load_preprocessing('train', info)
		# 	X=np.hstack((X,X2))
		# 	print X.shape

		print X.shape
		return X
	
	def load_train_X_separate(self):
		origin_instance=Load_origin_data(self.config)
		y=origin_instance.load_train_y()
		uid=origin_instance.load_train_uid()
		X=self.load_train_X()
		"""
		去除loginfo中沒有log
		"""
		# loginfo_instance=Loginfo(self.config)
		# info_idx,len_train,len_test=loginfo_instance.loginfo_idxs()
		# reader_idx=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(['Idx','ListingInfo']),encoding='utf-8')
		# train_info_idx=info_idx[:len_train]
		# train_info_idx=set(train_info_idx)
		# tmp_X=[]
		# tmp_uid=[]
		# tmp_y=[]
		# for i in range(len(reader_idx['Idx'])):
		# 	if reader_idx['Idx'][i] in train_info_idx:
		# 		tmp_X.append(X[i])
		# 		tmp_uid.append(uid[i])
		# 		tmp_y.append(y[i])

		# X=np.array(tmp_X)
		# uid=tmp_uid
		# y=tmp_y

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
		scale='numeric12_add_median'
		X=self.load_preprocessing('test', scale)
		print X.shape

		scale='category'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='location3'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='category_num2'
		X2=self.load_preprocessing('test', scale)
		X=np.hstack((X,X2))
		print X.shape

		scale='city_rank'
		X2=self.load_preprocessing('test', scale)
		X=np.hstack((X,X2))
		print X.shape

		scale='coor'
		X2=self.load_preprocessing('test', scale)
		X=np.hstack((X,X2))
		print X.shape

		scale='loginfo1'
		X2=self.load_preprocessing('test', scale)
		#X=X2
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='loginfo_limit1'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='loginfo_limit3'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='loginfo_limit7'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='loginfo3'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape	

		scale='updateinfo1'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='updateinfo_limit1'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='updateinfo_limit3'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='updateinfo_limit7'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		scale='updateinfo_time'
		X2=self.load_preprocessing('test', scale)
		X2=X2[:,1:]
		X=np.hstack((X,X2))
		print X.shape

		# scale='missing_scale'
		# X2=self.load_preprocessing('test', scale)
		# X=np.hstack((X,X2))
		# print X.shape	

		X_predict=X
		one_value_col=[]

		tmp_X=[]
		for i in range(len(X_predict[0])):
			if i not in self.one_value_col:
				tmp_X.append(X_predict[:,i])

		X=np.array(tmp_X).transpose()
		
		print X.shape
		return X
