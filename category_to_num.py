#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
import copy

class Category_to_num(object):
	def __init__(self,config):
		self.config=config
		self.ranklist_str=[]
		self.ranklist=[]
	def create_rank(self,n):
		self.find([i for i in range(n)],'')
		#print self.ranklist

	def find(self,l,r_list):
		if len(l)==0:
			if r_list not in self.ranklist_str:
				self.ranklist.append([int(j) for j in r_list.split(',')])
		else:
			for v in l:
				tmp_l=copy.deepcopy(l)
				tmp_l.remove(v)
				if r_list=='':		
					self.find(tmp_l,str(v))
				else:
					self.find(tmp_l,r_list+','+str(v))

	def analysis_feature(self,col,y):
		col_set=list(set(col))
		len_col=len(col_set)
		print "feature len:",len_col
		print col_set
		self.ranklist=[]
		self.ranklist_str=[]

		self.create_rank(len_col)
		ranklist=self.ranklist
		#print ranklist[0]
		best_dict={}
		best_col=[]
		best_cor=0
		for rank in ranklist:
			rank_dict={}
			for i,value in enumerate(col_set):
				rank_dict[value]=rank[i]

			rank_col=[]
			for value in col:
				rank_col.append(rank_dict[value])
			cor=np.corrcoef(rank_col,y)
			cor=cor[0,1]
			#print cor
			if abs(cor)>best_cor:
				best_cor=abs(cor)
				best_col=rank_col
				best_dict=rank_dict

		return best_cor,best_dict,best_col

	def load_feature(self):
		f='UserInfo'
		origin_instance=Load_origin_data(self.config)
		#features=origin_instance.load_feature(f)
		#features=['UserInfo_3','UserInfo_1','UserInfo_6','UserInfo_5','UserInfo_9','UserInfo_21','UserInfo_23','UserInfo_22','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15','UserInfo_16','UserInfo_17']
		
		features=[
			'UserInfo_3',
			'UserInfo_1',
			'UserInfo_6',
			'UserInfo_5',
			'UserInfo_9',
			'Education_Info1',
			'Education_Info2',
			'Education_Info3',
			'Education_Info4',
			'Education_Info5',
			'Education_Info6',
			'Education_Info7',
			'Education_Info8',
			#'WeblogInfo_20',
			'UserInfo_21',
			#'UserInfo_23',
			'UserInfo_22',
			'WeblogInfo_21',
			'SocialNetwork_1',
			'SocialNetwork_7',
			'SocialNetwork_12',
			'WeblogInfo_19',
			'UserInfo_11',
			'UserInfo_12',
			'UserInfo_13',
			'UserInfo_14',
			'UserInfo_15',
			'UserInfo_16',
			'UserInfo_17',
			'SocialNetwork_2'
		]

		reader1=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')
		y=origin_instance.load_train_y()
		l=[]

		for feature in features:
			print 'feature:',feature
			reader1[feature]=reader1[feature].apply(self._deal_nan)
			#cor=np.corrcoef(reader1[feature],y)
			#print cor
			#return
			best_cor,best_dict,best_col=self.analysis_feature(reader1[feature],y)
			print best_cor
			#print best_dict
			l.append(best_col)
			#return 

		l=np.array(l).transpose()
		pd.DataFrame(l).to_csv(self.config.path+"train/master_category_to_num.csv",seq=',',mode='wb',index=False,header=None)

	def _deal_nan(self,n):
		n2=n
		n=str(n)
		n=n.strip()

		if n=='nan':
			return -1
		else:
			return n2

def main():
	instance=Category_to_num(Config())
	instance.load_feature()
	pass

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')
	main()
