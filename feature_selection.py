#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from config import Config
from load_origin_data import Load_origin_data
from loginfo import Loginfo

class Feature_selection(object):
	def __init__(self,config):
		self.config=config

		self.origin_instance=Load_origin_data(config)
		self.train_uids=self.origin_instance.load_train_uid()
		len_train=len(self.train_uids)
		self.train_uids=np.array(self.train_uids)
		self.train_uids.shape=(len_train,1)

		self.predict_uids=self.origin_instance.load_predict_uid()
		len_predict=len(self.predict_uids)
		self.predict_uids=np.array(self.predict_uids)
		self.predict_uids.shape=(len_predict,1)

	def load_preprocessing(self,ftype,scale):
		"""
		读取预处理输出的特征文件
		"""
		X=pd.read_csv(self.config.path+ftype+'/master_'+scale+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		return np.array(X,dtype=float)


	def load_X(self,ftype):
		"""
		组合训练集多个特征文件
		"""
		#merge 2
		# dumps=['dumps_no_location','location'] #,'location'
		# infos=['loginfo1','loginfo_limit1','loginfo_limit3','loginfo_limit7','loginfo_time']
		# updates=['updateinfo1','updateinfo_limit1','updateinfo_limit3','updateinfo_limit7','updateinfo_time']
		# others=['category_num','city_rank','coor','category_weight','listingInfo_transform','ThirdParty_same_period','ThirdParty_same_section','ThirdParty_period_sum','ThirdParty_section_sum','missing_scale','ThirdParty_period_divide','ThirdParty_section_divide']
		#merge 3
		# dumps=['dumps_no_location'] #,'location'
		# infos=['loginfo1','loginfo_time']
		# updates=['updateinfo1','updateinfo_time']
		# others=['category_num','city_rank','coor','category_weight','listingInfo_transform','ThirdParty_same_period','ThirdParty_same_section','ThirdParty_period_sum','ThirdParty_section_sum','ThirdParty_period_divide','ThirdParty_section_divide','ThirdParty_row']
		
		#merge 4 单独对Thrid Party进行填充处理
		dumps=['ThirdParty_section_divide']#'ThirdParty_same_period','ThirdParty_same_section','ThirdParty_period_sum','ThirdParty_section_sum','ThirdParty_period_divide','ThirdParty_section_divide','ThirdParty_row'
		X=np.array([])
		for i,dump in enumerate(dumps):
			X2=self.load_preprocessing(ftype, dump)
			X2=X2[:,:len(X2[0])-1]
			print dump,X2.shape
			if i==0:
				X=X2
			else:
				X=np.hstack((X,X2))

		# for i,info in enumerate(infos):
		# 	X2=self.load_preprocessing(ftype, info)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print info,X2.shape
		# 	X=np.hstack((X,X2))

		# for i,update in enumerate(updates):
		# 	X2=self.load_preprocessing(ftype, update)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print update,X2.shape
		# 	X=np.hstack((X,X2))

		# for i,other in enumerate(others):
		# 	X2=self.load_preprocessing(ftype, other)
		# 	X2=X2[:,:len(X2[0])-1]
		# 	print other,X2.shape
		# 	X=np.hstack((X,X2))

		print X.shape
		return X

	def fill_missing_value(self):
		train_X=self.load_X('train')
		predict_X=self.load_X('test')
		len_train=len(train_X)

		X=np.vstack((train_X,predict_X))
		print X.shape
		m,n=X.shape
		l=[]
		split_features_num=0 #去除方差小的feature的个数
		for i in range(n):
			# nan转换为-1
			col=map(self._deal_nan,X[:,i])
			col=np.array(col)
			
			#方差<0.0001去除
			tmp_col=(np.array(col)-np.min(col))/(np.max(col)-np.min(col))
			tmp_col_var=np.var(tmp_col)
			if tmp_col_var<0.0001:
				split_features_num+=1
				continue

			# 6倍标准差外的值转换为mean+6倍标准差
			col2=col[np.where(col>=0)]
			self.col_mean=np.mean(col2)
			self.col_std=np.std(col2)*6+self.col_mean
			
			col=map(self._deal_std,col)
			per=float(len(col2))/float(m)
			# 缺失值小于20%的值，填充为均值
			if per>0.85:
				col=map(self._deal_fill,col)
			
			#col=map(self._deal_round,col)
			#if self.is_choose_col(l, col):
			l.append(col)

			if i%10==0:
				print i

		print 'split feature: ',split_features_num
		X=np.array(l).transpose()

		X_train=X[:len_train]
		X_train=np.hstack((self.train_uids,X_train))

		X_predict=X[len_train:]
		X_predict=np.hstack((self.predict_uids,X_predict))
		print X.shape
		return X_train,X_predict

	def is_choose_col(self,l,new_col):
		"""
		判断当前列与已有列之间的相关性，对相关性高的列不加入l中
		"""
		for col in l:
			cor=np.corrcoef(col,new_col)
			if cor[0,1]>0.99:
				return False
		return True

	def run_selection(self):
		params={
			'min_cols':100, #最小的列数
			'max_iter':200, #最大比较轮数
			'max_no_change_iter':20,#不产生结果的最大论数
			'min_sim':0.99, #相似度大于多少列作则一选择
			'seed':1, #随机种子
			'slient':False
		}
		train_X=self.load_X('train')
		predict_X=self.load_X('test')
		len_train=len(train_X)

		X=np.vstack((train_X,predict_X))
		X=self.col_selection(X, params)
		X_train=X[:len_train]
		X_train=np.hstack((self.train_uids,X_train))

		X_predict=X[len_train:]
		X_predict=np.hstack((self.predict_uids,X_predict))
		print X.shape
		return X_train,X_predict

	def output_selection(self,X_train,X_predict):
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_ThirdParty_section_divide_selection.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_ThirdParty_section_divide_selection.csv",seq=',',mode='wb',index=False,header=None)

	def col_selection(self,X,params):
		"""
		params={
			'min_cols':0, #最小的列数
			'max_iter':0, #最大比较轮数
			'max_no_change_iter':,#不产生结果的最大论数
			'min_sim':0.99, #相似度大于多少列作则一选择
			'seed':, #随机种子,
			'slient':False
		}
		
		"""
		random.seed(params['seed'])

		no_change_iter=0
		m,n=X.shape
		last_n=n+1
		for i in range(params['max_iter']):		
			m,n=X.shape
			if n<=params['min_cols']:
				break
			if last_n==n:
				no_change_iter+=1
			if no_change_iter>=params['max_no_change_iter']:
				break
			X=self._deal_col_selection(X, params)
			last_n=n
			if not params['slient']:
				print 'iter: ',i,' col num: ',n,' no_change_iter: ',no_change_iter

		return X

	def _deal_col_selection(self,X,params):
		m,n=X.shape
		indexes=[i for i in range(n)]
		random.shuffle(indexes)
		X=X[:,indexes]

		l=[]
		lastChoose=False
		for i in range(n-1):
			isChooseOne=self.is_choose_one(X[:,i],X[:,i+1],params['min_sim'])
			if isChooseOne:
				lastChoose=True
			else:
				l.append(i)
				lastChoose=False
		
		if n-1 not in l:
			l.append(n-1)


		X=X[:,l]
		return X

	def is_choose_one(self,col1,col2,sim):
		cor=np.corrcoef(col1,col2)
		if cor[0,1]>sim:
			#print 'bingo'
			return True
		else:
			return False


	def _deal_round(self,n):
		return round(n,2)

	def output_analysis(self,X_train,X_predict):
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_merge_thirdparty.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_merge_thirdparty.csv",seq=',',mode='wb',index=False,header=None)

	def _deal_nan(self,n):
		if str(n)=='nan':
			return -1
		else:
			return n

	def _deal_std(self,n):
		if n>self.col_std:
			return round(self.col_std,2)
		else:
			return round(n,2)

	def _deal_fill(self,n):
		if n==-1:
			return round(self.col_mean,2)
		return round(n,2)

def main():
		instance=Feature_selection(Config())
		#X_train,X_predict=instance.fill_missing_value()
		#instance.output_analysis(X_train, X_predict)
		
		X_train,X_predict=instance.run_selection()
		instance.output_selection(X_train, X_predict)

if __name__ == '__main__':
	main()