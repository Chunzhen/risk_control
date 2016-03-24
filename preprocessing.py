#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
import json
import time

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
		return features_category,features_numeric,reader_category,reader_numeric,len_train,len_predict

	def dumps_scale(self):
		features_category,features_numeric,reader_category,reader_numeric,len_train,len_predict=self.load_data()
		i=0
		dumps=pd.DataFrame()
		for feature in features_category:
			if feature=='UserInfo_24':
				continue
				reader_category[feature]=reader_category[feature].apply(self._deal_userInfo_24)
			elif feature=='UserInfo_2' or feature=='UserInfo_7' or feature=='UserInfo_4' or feature=='UserInfo_8' or feature=='UserInfo_9' or feature=='UserInfo_20' or feature=='UserInfo_19':
				continue
				reader_category[feature]=reader_category[feature].apply(self._deal_userinfo_normal)
			else:
				reader_category[feature]=reader_category[feature].apply(self._deal_nan)
			#col=reader_category[feature].tolist()
			#print feature
			#col=set(col)
			tmp_dummys=pd.get_dummies(reader_category[feature])
			#print tmp_dummys.shape
			
			if i==0:
				dumps=tmp_dummys
				i+=1
			else:
				dumps=np.hstack((dumps,tmp_dummys))
				i+=1

		print dumps.shape

		dumps_numeric=pd.DataFrame()
		i=0
		for feature in features_numeric:	
			if feature=='ListingInfo':
				reader_numeric[feature]=reader_numeric[feature].apply(self._deal_date)
			else:
				reader_numeric[feature]=reader_numeric[feature].apply(self._deal_nan_digit)
			col=reader_numeric[feature].tolist()
			col=set(col)
			#print col
			if len(col)<10:
				tmp_dummys=pd.get_dummies(reader_numeric[feature])
				if i==0:
					dumps_numeric=tmp_dummys
				else:
					dumps_numeric=np.hstack((dumps_numeric,tmp_dummys))
				i+=1

		print dumps_numeric.shape

		X=np.hstack((reader_numeric,dumps))
		X=np.hstack((X,dumps_numeric))
		print X.shape
		X_train=X[:len_train]
		X_predict=X[len_train:]
		
		return X_train,X_predict

	def output_dumps_scale(self):
		X_train,X_predict=self.dumps_scale()
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_dumps2.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_dumps2.csv",seq=',',mode='wb',index=False,header=None)

	def _deal_date(self,n):
		try:
			t=time.strptime(str(n),"%Y/%m/%d")
		except:
			t=time.strptime(str(n),"%d/%m/%Y")
		return (time.mktime(t)-time.mktime(time.strptime("1/1/2010","%d/%m/%Y")))/100

	def _deal_nan_digit(slef,n):
		if str(n)=='nan':
			return -1
		else:
			return n

	def _deal_nan(self,n):
		n=str(n)
		n=n.strip()

		if n=='nan':
			return -1
		else:
			return n

	def _deal_userinfo_normal(self,n):
		n=str(n)
		n=n.strip()
		index=n.replace(u"市",'')
		
		if n=='nan':
			return -1
		else:
			return n

	def _deal_userInfo_24(self,n):
		n=str(n)
		n=n.strip()
		index=n.find(u"市")
		
		if index>0:
			n=n[:index]
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

	def location_scale(self):
		features_category,features_numeric,reader_category,reader_numeric,len_train,len_predict=self.load_data()
		i=0
		dumps=pd.DataFrame()
		for feature in features_category:
			if feature=='UserInfo_24' or feature=='UserInfo_2' or feature=='UserInfo_7' or feature=='UserInfo_4' or feature=='UserInfo_8' or feature=='UserInfo_20' or feature=='UserInfo_19':
				#print feature
				self.location=self.load_location_json(feature)
				reader_province=reader_category[feature].apply(self._deal_province_scale)
				reader_city=reader_category[feature].apply(self._deal_city_scale)
				#break
			else:
				continue

			#tmp_dummys=pd.get_dummies(reader_province)
			#print tmp_dummys.shape
			tmp_dummys=pd.get_dummies(reader_city)
			
			if i==0:
				#dumps=np.hstack((tmp_dummys,tmp_dummys2))
				dumps=tmp_dummys
				i+=1
			else:
				#dumps=np.hstack((dumps,tmp_dummys,tmp_dummys2))
				dumps=np.hstack((dumps,tmp_dummys))
				i+=1

		print dumps.shape
		X=dumps
		X_train=X[:len_train]
		X_predict=X[len_train:]
		return X_train,X_predict

	def output_location_scale(self):
		X_train,X_predict=self.location_scale()
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_location.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_location.csv",seq=',',mode='wb',index=False,header=None)

	def _deal_province_scale(self,n):	
		try:
			location=self.location[n]
			return location[0]
		except:
			return 'nan'

	def _deal_city_scale(self,n):
		try:
			location=self.location[n]
			return location[1]
		except:
			return 'nan'
		

	def load_location_json(self,feature):
		f=open(self.config.path_location+feature+'.json')
		lines=f.readline()
		s=json.loads(u''+lines)
		return s

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

	



