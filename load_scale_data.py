#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import PCA,KernelPCA

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
		dumps=['dumps_no_location'] #,'location'
		#infos=['loginfo1','loginfo_limit1','loginfo_limit3','loginfo_limit7','loginfo_time']
		#updates=['updateinfo1','updateinfo_limit1','updateinfo_limit3','updateinfo_limit7','updateinfo_time']
		others=['category_num','city_rank','coor','category_weight','listingInfo_transform','ThirdParty_same_period','ThirdParty_same_section','ThirdParty_period_sum','ThirdParty_section_sum','ThirdParty_period_divide','ThirdParty_section_divide','ThirdParty_row','weblog_row']
		X=np.array([])
		# for i,dump in enumerate(dumps):
		# 	X2=self.load_preprocessing('train', dump)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print dump,X2.shape
		# 	if i==0:
		# 		X=X2
		# 	else:
		# 		X=np.hstack((X,X2))

		# for i,info in enumerate(infos):
		# 	X2=self.load_preprocessing('train', info)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print info,X2.shape
		# 	if i==0:
		# 		X=X2
		# 	else:
		# 		X=np.hstack((X,X2))

		# for i,update in enumerate(updates):
		# 	X2=self.load_preprocessing('train', update)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print update,X2.shape
		# 	X=np.hstack((X,X2))

		# for i,other in enumerate(others):
		# 	X2=self.load_preprocessing('train', other)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print other,X2.shape
		# 	X=np.hstack((X,X2))

		# scale='ThirdParty_part'
		# X2=self.load_preprocessing('train', scale)
		# #X=X2
		# X=X2[:,:len(X2[0])-1]
		# print X.shape

		scale='ThirdParty_same_section'
		X2=self.load_preprocessing('train', scale)
		#X=X2
		X=X2[:,:len(X2[0])-1]
		#X=np.hstack((X,X2))
		print scale, X.shape


		# scale='merge_thirdparty'
		# X2=self.load_preprocessing('train', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		# scale='ThirdParty_period_sum'
		# X2=self.load_preprocessing('train', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		# scale='ThirdParty_section_sum'
		# X2=self.load_preprocessing('train', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		#X=np.nan_to_num(X)
		#indexs=self.load_feature_importance()
		#X=X[:,indexs]
		X=self.min_max_scale(X)
		#X=self.pca_salce(X,100)
		print X.shape
		return X
	
	def pca_salce(self,X,n):
		pca=PCA(n_components=n)
		#pca = KernelPCA(kernel="linear", fit_inverse_transform=True, gamma=10)
		return pca.fit_transform(X)

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
		dumps=['dumps_no_location','location']
		infos=['loginfo1','loginfo_limit1','loginfo_limit3','loginfo_limit7','loginfo_time']
		updates=['updateinfo1','updateinfo_limit1','updateinfo_limit3','updateinfo_limit7','updateinfo_time']
		others=['category_num','city_rank','coor','category_weight','Userinfo_weight']
		X=np.array([])
		# for i,dump in enumerate(dumps):
		# 	X2=self.load_preprocessing('test', dump)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print dump,X2.shape
		# 	if i==0:
		# 		X=X2
		# 	else:
		# 		X=np.hstack((X,X2))

		# for i,info in enumerate(infos):
		# 	X2=self.load_preprocessing('test', info)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print info,X2.shape
		# 	if i==0:
		# 		X=X2
		# 	else:
		# 		X=np.hstack((X,X2))

		# for i,update in enumerate(updates):
		# 	X2=self.load_preprocessing('test', update)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print update,X2.shape
		# # 	X=np.hstack((X,X2))

		# for i,other in enumerate(others):
		# 	X2=self.load_preprocessing('test', other)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print other,X2.shape
		# 	X=np.hstack((X,X2))


		# scale='ThirdParty_same_period'
		# X2=self.load_preprocessing('test', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		# scale='ThirdParty_same_period_exp'
		# X2=self.load_preprocessing('test', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		# scale='ThirdParty_same_period_square'
		# X2=self.load_preprocessing('test', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		# scale='ThirdParty_same_section'
		# X2=self.load_preprocessing('test', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		# scale='ThirdParty_same_section_exp'
		# X2=self.load_preprocessing('test', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape

		# scale='ThirdParty_same_section_square'
		# X2=self.load_preprocessing('test', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale, X.shape
		scale='feature_467'
		X2=self.load_preprocessing('test', scale)
		#X=X2
		X=X2[:,:len(X2[0])-1]
		#X=np.hstack((X,X2))
		print scale, X.shape


		scale='merge_thirdparty'
		X2=self.load_preprocessing('test', scale)
		#X=X2
		X2=X2[:,:len(X2[0])-1]
		X=np.hstack((X,X2))
		print scale, X.shape

		# scale='ThirdParty_row'
		# X2=self.load_preprocessing('test', scale)
		# #X=X2
		# X2=X2[:,:len(X2[0])-1]
		# X=np.hstack((X,X2))
		# print scale,X.shape

		# X_predict=X
		# one_value_col=[]

		# tmp_X=[]
		# for i in range(len(X_predict[0])):
		# 	if i not in self.one_value_col:
		# 		tmp_X.append(X_predict[:,i])

		# X=np.array(tmp_X).transpose()
		#X=np.nan_to_num(X)
		#X=self.normalizer_scale(X)
		#indexs=self.load_feature_importance()
		#X=X[:,indexs]

		print X.shape
		return X

	def importance_scale(self,X,indexs):
		return X[:,indexs]

	def load_feature_importance(self):
		X=pd.read_csv(self.config.path+'train/features_importance_merge3.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		indexs=[]
		for i in range(len(X[0])):
			if X[1][i]>1:
				indexs.append(X[0][i])
		return indexs

	def standard_scale(self,X):
		"""
		:type X: numpy.array 特征矩阵
		:rtype X: numpy.array 变换后特征
		:特征变换工具函数
		"""
		scaler=StandardScaler()
		return scaler.fit_transform(X)

	def min_max_scale(self,X):
		scaler=MinMaxScaler()
		return scaler.fit_transform(X)

	def normalizer_scale(self,X):
		scaler=Normalizer()
		return scaler.fit_transform(X)

	#每个feature的中位数
	def median_feature(self,X):
		m,n=X.shape
		X_median=[]
		for i in range(n):
			median=np.median(X[:,i])
			X_median.append(median)
		return X_median

	#median 填充-1值
	def fill_scale(self,X,X_median):
		m,n=X.shape
		for i in range(m):
			for j in range(n):
				if X[i][j]==-1 or X[i][j]==-2:
					X[i][j]=X_median[j]
		return X

	def log_scale(self,X):
		m,n=X.shape
		for i in range(m):
			for j in range(n):
				if X[i][j]>0:
					X[i][j]=math.log10(X[i][j])
		return X

	def log_scale_move(self,X):
		n,m=X.shape
		for i in range(m):
			column=X[:,i]

			c_max=np.max(column)
			c_min=np.min(column)
			for j in range(n):
				column[j]=math.log10(column[j]-c_min+1)

			X[:,i]=column
		return X
