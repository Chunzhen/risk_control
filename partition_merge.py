#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

from config import Config
from load_origin_data import Load_origin_data

from sklearn import metrics

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class Partition_merge(object):
	def __init__(self,config):
		self.config=config

	def load_clf_file(self,level,name):
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		#print reader
		d={}
		for i in range(len(reader[0])):
			d[reader[0][i]]=reader[1][i]
		return d

	def level_data(self):
		level='one'
		clf_name='xgb2000'
		clf_name2='gbdt20'
		#读取验证集数据
		load_data_instance=Load_origin_data(self.config)
			
		uid=load_data_instance.load_train_uid()
		y=load_data_instance.load_train_y()
		test_uid_0=[]
		test_uid_1=[]
		for i in range(len(y)):
			if y[i]==0:
				test_uid_0.append(uid[i])
			else:
				test_uid_1.append(uid[i])

		test_uid_0=set(test_uid_0)
		test_uid_1=set(test_uid_1)

		prob=[]
		real=[]
		prob_1=[]
		prob_0=[]

		#读取某分类器预测结果
		column_dict=self.load_clf_file(level,clf_name)
		#排序
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])


		column_ranks=self.level_ranks(level,clf_name)
		column_ranks2=self.level_ranks(level,clf_name2)

		i=0
		one_diff=[]
		zero_diff=[]
		one_index=[]
		zero_index=[]

		for uid, score in column_dict2:
			diff=column_ranks[uid][0]-column_ranks2[uid][0]
			if uid in test_uid_0:
				zero_diff.append(diff)
				zero_index.append(i)
			else:
				one_diff.append(diff)
				one_index.append(i)
				pass
				
			i+=1

		#计算blend后的AUC
		# for uid,score in column_dict.items():
		# 	prob.append(score)
		# 	if uid in test_uid_0:
		# 		real.append(0)
		# 		prob_0.append(score)
		# 	elif uid in test_uid_1:
		# 		real.append(1)
		# 		prob_1.append(score)
		# 	else:
		# 		print "error"

		# auc_score=metrics.roc_auc_score(real,prob)
		# print name,"  "," auc:",auc_score	
		# print '0:',max(prob_0),min(prob_0)
		# print "1:",max(prob_1),min(prob_1)

		#绘制不同分类器的排名差结果
		idex=0
		self.print_diff(zero_diff[idex:],zero_index[idex:],one_diff[idex:],one_index[idex:])
		return

	def print_diff(self,zero_diff,zero_index,one_diff,one_index):
		"""
		:type zero_diff: List[int] 0类的排名分布差
		:type zero_index: List[int] 0类的下标
		:type one_diff: List[int] 1类的排名分布差
		:type one_index: List[int] 1类的下标
		"""
		plt.scatter(zero_index,zero_diff)
		plt.scatter(one_index,one_diff,c='red')
		plt.show()

	def level_ranks(self,level,name):
		"""
		返回不同分类样本在本分类器中的排名
		"""
		ranks={}
		column_dict=self.load_clf_file(level,name)
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])
		i=0

		for uid, score in column_dict2:
			rank=ranks.get(uid,[])
			rank.append(i)
			ranks[uid]=rank
			i+=1

		return ranks

def main():
	instance=Partition_merge(Config())
	instance.level_data()
	pass

if __name__ == '__main__':
	main()