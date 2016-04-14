#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
from loginfo import Loginfo
from updateinfo import Updateinfo
import threading
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

class Local_auc(object):
	def __init__(self,config):
		self.config=config
		self.uids=Load_origin_data(config).load_train_uid()
		#self.predict_uids=Load_origin_data(config).load_predict_uid()
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

	def load_predict_file(self,level,name):
		reader=pd.read_csv(self.config.path_predict+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8')
		reader=np.array(reader)
		#print reader
		d={}
		for i in range(len(reader)):
			d[str(int(reader[i,0]))]=reader[i,1]

		return d

	def load_daily_file(self):
		reader=pd.read_csv(self.config.path_verify+'daily_test.csv',iterator=False,delimiter=',',encoding='utf-8')
		return np.array(reader)

	def load_final_file(self):
		reader=pd.read_csv(self.config.path_verify+'final_test.csv',iterator=False,delimiter=',',encoding='utf-8')
		return np.array(reader)	

	def output_test_target(self):
		predict_uids=Load_origin_data(self.config).load_predict_uid()

		daily_list=self.load_daily_file()

		d_daily={}
		for i in range(len(daily_list[:,0])):
			d_daily[str(int(daily_list[i,0]))]=daily_list[i,1]


		final_list=self.load_final_file()

		d_final={}
		for i in range(len(final_list[:,0])):
			d_final[str(int(final_list[i,0]))]=final_list[i,1]

		targets=[]
		for uid in predict_uids:
			if str(uid) in d_daily:
				targets.append(d_daily[str(uid)])
			else:
				targets.append(d_final[str(uid)])

		f=open(self.config.path+'test_target.csv','wb')
		f.write('Idx,target\n')
		for i,uid in enumerate(predict_uids):
			f.write(str(uid)+','+str(targets[i])+'\n')
		f.close()

	def get_predict_list(self,uids,predicts):
		predicts_list=[]
		for uid in uids:
			predicts_list.append(predicts[str(int(uid))])
		return (np.array(predicts_list)--np.min(predicts_list))/(np.max(predicts_list)-np.min(predicts_list))

	def test_verify(self):
		level='one'
		name='lr' 
		#name='merge'
		daily_test=self.load_daily_file()
		final_test=self.load_final_file()

		predicts=self.load_predict_file(level,name)
		predicts2=self.load_predict_file(level,'xgb900_master_467_third_party')
		predicts3=self.load_predict_file(level,'3-31-max')#3-31-max
		#print predicts

		predicts_list=self.get_predict_list(daily_test[:,0],predicts)
		predicts_list2=self.get_predict_list(daily_test[:,0],predicts2)
		predicts_list3=self.get_predict_list(daily_test[:,0],predicts3)


		#predicts_list=predicts_list*0.4+predicts_list2*1.5+predicts_list3*1
		predicts_list=predicts_list*0.0+predicts_list2*1.5+predicts_list3*0

		auc_score=metrics.roc_auc_score(daily_test[:,1],predicts_list)
		print 'daily test: ',name,' ',auc_score

		predicts_list=self.get_predict_list(final_test[:,0],predicts)
		predicts_list2=self.get_predict_list(final_test[:,0],predicts2)
		predicts_list3=self.get_predict_list(final_test[:,0],predicts3)

		predicts_list=predicts_list*0+predicts_list2*1.5+predicts_list3*0
		auc_score=metrics.roc_auc_score(final_test[:,1],predicts_list)
		print 'final_test: ',name,' ',auc_score

	def daily_final_log_diff(self):
		daily_test=self.load_daily_file()
		final_test=self.load_final_file()
		info_instance=Loginfo(self.config)
		Idxs,len_train,len_test=info_instance.loginfo_idxs()
		Idxs=set(Idxs)

		daily_test_Idxs=set(daily_test[:,0])
		final_test_Idxs=set(final_test[:,0])

		len_daily=len(daily_test_Idxs)
		len_final=len(final_test_Idxs)

		print len(daily_test_Idxs)-len(Idxs&daily_test_Idxs)
		print len(final_test_Idxs)-len(Idxs&final_test_Idxs)


def main():
	instance=Local_auc(Config())
	instance.test_verify()
	#instance.daily_final_log_diff()
	pass

if __name__ == '__main__':
	main()