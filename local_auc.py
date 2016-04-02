#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
import threading
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

class Local_auc(object):
	def __init__(self,config):
		self.config=config
		self.uids=Load_origin_data(config).load_train_uid()
		self.y=Load_origin_data(config).load_train_y()
		pass

	def load_clf_file(self,level,name):
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		reader=np.array(reader)
		d={}
		for i in range(len(reader)):
			d[str(int(reader[i,0]))]=reader[i,1]
		return d
	def level_ranks(self,level,name):
		"""
		返回不同分类样本在本分类器中的排名
		"""
		ranks={}
		column_dict=self.load_clf_file(level,name)
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])
		i=0
		l=[]
		for uid, score in column_dict2:
			ranks[uid]=float(i)
			i+=1
		#print ranks
		for uid in self.uids:
			l.append(ranks[str(uid)])
		l2=np.array(l)-np.min(l)/(np.max(l)-np.min(l))
		auc_score=metrics.roc_auc_score(self.y,l2)
		print name,' ',auc_score
		return np.array(l)

	def load_file(self,level,name):
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		reader=np.array(reader)
		#print reader
		d={}
		for i in range(len(reader)):
			d[str(int(reader[i,0]))]=reader[i,1]

		l=[]
		for uid in self.uids:
			l.append(d[str(uid)])

		auc_score=metrics.roc_auc_score(self.y,l)
		print name,' ',auc_score
		return np.array(l)-np.min(l)/(np.max(l)-np.min(l))

	def merge(self):
		level='one/old'
		names=[
		# 'gbdt20',
		# 'rf100',
		# 'xgb2000_dumps_location',
		# 'xgb2000_dumps_no_location_category_num_cityrank_loginfo1',
		'xgb1000_master_x',
		# 'xgb1000_dumps',
		'xgb1000_dumps_location',
		'lr_x',
		'lr_sag',
		'xgb1000_master_lr',
		'xgb2000_master_lr',
		'xgb1000_master_dumps_no_location'
		# 'xgb1000',
		# 'ada100',
		# 'ada20'
		]
		l=[]
		for name in names:
			x=self.level_ranks(level, name)
			l.append(x)

		pred=0.5*l[0]+0.5*l[1]+0.5*l[2]


		auc_score=metrics.roc_auc_score(self.y,pred)
		print 'merge auc:',auc_score
		pass

	def output(self):
		level='one'
		names=[
		'gbdt20',
		'rf100',
		'xgb2000_dumps_location',
		'xgb2000_dumps_no_location_category_num_cityrank_loginfo1',
		'xgb1000_master_x',
		'xgb1000_dumps',
		'xgb1000_dumps_location',
		'lr_sag',
		'xgb1000',
		'ada100',
		'ada20'
		]
		l=[]
		for name in names:
			x=self.load_file(level,name)
			l.append(x)

		X=np.array(l).transpose()
		pd.DataFrame(X).to_csv(self.config.path+"train/master_merge.csv",seq=',',mode='wb',index=False,header=None)

def main():
	instance=Local_auc(Config())
	instance.merge()
	pass

if __name__ == '__main__':
	main()