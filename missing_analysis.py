#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
from load_scale_data import Load_scale_data
from loginfo import Loginfo
from updateinfo import Updateinfo
import json
import time

class Missing_analysis(object):
	def __init__(self,config):
		self.config=config

	def load_data(self):
		scale_instance=Load_scale_data(self.config)
		origin_instance=Load_origin_data(self.config)
		y=origin_instance.load_train_y()
		feature_types=['UserInfo','WeblogInfo','Education_Info','ThirdParty','SocialNetwork']
		features=origin_instance.load_feature(feature_types[4])

		reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')
		#print reader
		#return
		for feature in features:
			reader[feature]=reader[feature].apply(self.is_missing)

		reader=np.array(reader)
		#print reader

		missing_1=[]
		missing_0=[]
		for i in range(len(y)):
			if y[i]==0:
				missing_0.append(sum(reader[i]))
			else:
				missing_1.append(sum(reader[i]))

		print np.max(missing_1),' ',np.min(missing_1),' ',np.mean(missing_1),' ',np.median(missing_1)
		print np.max(missing_0),' ',np.min(missing_0),' ',np.mean(missing_0),' ',np.median(missing_0)

	def info_missing(self):
		origin_instance=Load_origin_data(self.config)
		loginfo_instance=Loginfo(self.config)
		info_idx,len_train,len_test=loginfo_instance.loginfo_idxs()
		train_info_idx=info_idx[:len_train]
		y=origin_instance.load_train_y()
		reader_idx=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(['Idx','ListingInfo']),encoding='utf-8')
		
		train_info_idx=set(train_info_idx)
		missing_1=0
		missing_0=0
		trading_time=[]
		for i in range(len(reader_idx['Idx'])):
			if reader_idx['Idx'][i] not in train_info_idx:
				if y[i]==0:
					missing_0+=1
					print '0: ',reader_idx['ListingInfo'][i]
				else:
					missing_1+=1
					#print '1: ',reader_idx['ListingInfo'][i]
				trading_time.append(reader_idx['ListingInfo'][i])

		print 'missing_0:',missing_0
		print 'missing_1:',missing_1
		#print trading_time

	def updateinfo_missing(self):
		origin_instance=Load_origin_data(self.config)
		loginfo_instance=Updateinfo(self.config)
		info_idx,len_train,len_test=loginfo_instance.loginfo_idxs()
		train_info_idx=info_idx[:len_train]
		y=origin_instance.load_train_y()

		reader_idx=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(['Idx','ListingInfo']),encoding='utf-8')
		
		train_info_idx=set(train_info_idx)
		missing_1=0
		missing_0=0
		trading_time=[]
		for i in range(len(reader_idx['Idx'])):
			if reader_idx['Idx'][i] not in train_info_idx:
				if y[i]==0:
					missing_0+=1
					#print '0: ',reader_idx['ListingInfo'][i]
				else:
					missing_1+=1
					print '1: ',reader_idx['ListingInfo'][i]
				trading_time.append(reader_idx['ListingInfo'][i])

		print 'missing_0:',missing_0
		print 'missing_1:',missing_1
		print trading_time


	def is_missing(self,n):
		if str(n)==u'nan': # or str(n)==u'-1' or str(n)==u'不详'
			return 1
		else:
			return 0
		

def main():
	instance=Missing_analysis(Config())
	instance.info_missing()
	pass

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')
	main()
