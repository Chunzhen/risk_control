#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
import copy

class Log_analysis(object):
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

	def analysis(self):
		f='ThirdParty'
		origin_instance=self.origin_instance
		features=origin_instance.load_feature(f)
		print len(features)

		reader1=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')
		reader2=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')

		len_train=len(reader1)
		len_test=len(reader2)

		reader=pd.concat([reader1,reader2],ignore_index=True)
		
		for feature in features:
			self.feature_min=np.min(reader[feature])
			reader[feature]=reader[feature].apply(self._deal_move)

		print "bingo feature move..."
		#同一时期，不同阶段的特征作差处理
		l=[]
		# for i in range(1,8):
		# 	for j in range(1,18):
		# 		for k in range(j+1,18):
		# 			#l.append(reader['ThirdParty_Info_Period'+str(i)+'_'+str(j)]**2-reader['ThirdParty_Info_Period'+str(i)+'_'+str(k)]**2)
		# 			l.append(reader['ThirdParty_Info_Period'+str(i)+'_'+str(j)]-reader['ThirdParty_Info_Period'+str(i)+'_'+str(k)])
					#l.append(np.exp(-reader['ThirdParty_Info_Period'+str(i)+'_'+str(j)])-np.exp(-reader['ThirdParty_Info_Period'+str(i)+'_'+str(k)]))

		#不同时期，相同阶段的特征作差处理
		for i in range(1,18):
			for j in range(1,8):
				for k in range(j+1,8):
					#print i,' ',j,' ',k
					#l.append(np.exp(-reader['ThirdParty_Info_Period'+str(j)+'_'+str(i)])-np.exp(-reader['ThirdParty_Info_Period'+str(k)+'_'+str(i)]))
					#l.append(reader['ThirdParty_Info_Period'+str(j)+'_'+str(i)]**2-reader['ThirdParty_Info_Period'+str(k)+'_'+str(i)]**2)
					l.append(reader['ThirdParty_Info_Period'+str(j)+'_'+str(i)]-reader['ThirdParty_Info_Period'+str(k)+'_'+str(i)])

		X=np.array(l).transpose()

		#X=np.hstack((np.array(reader),l))

		X_train=X[:len_train]
		X_train=np.hstack((self.train_uids,X_train))

		X_predict=X[len_train:]
		X_predict=np.hstack((self.predict_uids,X_predict))
		print X.shape
		return X_train,X_predict,f

	def analysis2(self):
		f='ThirdParty'
		origin_instance=self.origin_instance
		features=origin_instance.load_feature(f)
		print len(features)

		reader1=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')
		reader2=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')

		len_train=len(reader1)
		len_test=len(reader2)

		reader=pd.concat([reader1,reader2],ignore_index=True)
		
		for feature in features:
			self.feature_min=np.min(reader[feature])
			reader[feature]=reader[feature].apply(self._deal_move)

		print "bingo feature move..."
		#同一时期，不同阶段的特征累积
		l=[]
		# for i in range(1,8):
		# 	for j in range(1,17):
		# 			reader['ThirdParty_Info_Period'+str(i)+'_'+str(j+1)]=reader['ThirdParty_Info_Period'+str(i)+'_'+str(j)]+reader['ThirdParty_Info_Period'+str(i)+'_'+str(j+1)]
		# 			l.append(reader['ThirdParty_Info_Period'+str(i)+'_'+str(j+1)])

		#不同时期，相同阶段的特征累积
		for i in range(1,18):
			for j in range(1,7):
					reader['ThirdParty_Info_Period'+str(j+1)+'_'+str(i)]=reader['ThirdParty_Info_Period'+str(j)+'_'+str(i)]+reader['ThirdParty_Info_Period'+str(j+1)+'_'+str(i)]
					l.append(reader['ThirdParty_Info_Period'+str(j+1)+'_'+str(i)])

		X=np.array(l).transpose()

		#X=np.hstack((np.array(reader),l))

		X_train=X[:len_train]
		X_train=np.hstack((self.train_uids,X_train))

		X_predict=X[len_train:]
		X_predict=np.hstack((self.predict_uids,X_predict))
		print X.shape
		return X_train,X_predict,f

	def analysis3(self):
		"""
		特征相除
		"""
		f='ThirdParty'
		origin_instance=self.origin_instance
		features=origin_instance.load_feature(f)
		print len(features)

		reader1=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')
		reader2=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')

		len_train=len(reader1)
		len_test=len(reader2)

		reader=pd.concat([reader1,reader2],ignore_index=True)
		
		for feature in features:
			#self.feature_min=np.min(reader[feature])
			reader[feature]=reader[feature].apply(self._deal_nan)

		print "bingo feature nan..."
		#同一时期，不同阶段的特征作除处理
		l=[]
		a=0
		# for i in range(1,8):
		# 	for j in range(1,18):
		# 		for k in range(j+1,18):
		# 			l.append(map(self._deal_divide,reader['ThirdParty_Info_Period'+str(i)+'_'+str(j)],reader['ThirdParty_Info_Period'+str(i)+'_'+str(k)]))
		# 			a+=1
		# 			print 'bingo:',a 


		#不同时期，相同阶段的特征作除处理
		for i in range(1,18):
			for j in range(1,8):
				for k in range(j+1,8):
					l.append(map(self._deal_divide,reader['ThirdParty_Info_Period'+str(j)+'_'+str(i)],reader['ThirdParty_Info_Period'+str(k)+'_'+str(i)]))
					a+=1
					print 'bingo:',a 

		X=np.array(l).transpose()

		#X=np.hstack((np.array(reader),l))

		X_train=X[:len_train]
		X_train=np.hstack((self.train_uids,X_train))

		X_predict=X[len_train:]
		X_predict=np.hstack((self.predict_uids,X_predict))
		print X.shape
		return X_train,X_predict,f

	def _deal_divide(self,a,b):
		if a==-1 and b==-1:
			return -8
		elif a==-1 and b==0:
			return -7
		elif a==-1 and b>0:
			return -6
		elif a==0 and b==-1:
			return -5
		elif a>0 and b==-1:
			return -4
		elif a==0 and b==0:
			return -3
		elif a==0 and b>0:
			return -2
		elif a>0 and b==0:
			return -1
		else:
			return round(float(a)/float(b),4)

	def _deal_move(self,n):
		if str(n)=='nan':
			return -1-self.feature_min+1
		else:
			return n-self.feature_min+1

	def _deal_nan(self,n):
		if str(n)=='nan':
			return -1
		else:
			return n

	def output_analysis(self,X_train,X_predict,f):
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_"+f+"_section_sum.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_"+f+"_section_sum.csv",seq=',',mode='wb',index=False,header=None)
		pass

	def output_analysis3(self,X_train,X_predict,f):
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_"+f+"_section_divide.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_"+f+"_section_divide.csv",seq=',',mode='wb',index=False,header=None)
		pass

def main():
	instance=Log_analysis(Config())
	X_train,X_predict,f=instance.analysis3()
	instance.output_analysis3(X_train, X_predict,f)
	pass

if __name__ == '__main__':
	main()
