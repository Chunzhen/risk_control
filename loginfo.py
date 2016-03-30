#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
import time
import copy
import json

class Loginfo(object):
	def __init__(self,config):
		self.config=config

	def load_info(self):
		reader_train=pd.read_csv(self.config.path_origin_train_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		reader_test=pd.read_csv(self.config.path_origin_predict_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		len_train=len(reader_train)
		len_test=len(reader_test)
		reader=pd.concat([reader_train,reader_test],ignore_index=True)
		# reader['Listinginfo1']=reader['Listinginfo1'].apply(self._deal_date)
		# reader['LogInfo3']=reader['LogInfo3'].apply(self._deal_date)
		Idxs=self.get_Idx(reader['Idx'])
		#print Idxs
		codes=list(set(reader['LogInfo1']))
		types=list(set(reader['LogInfo2']))

		d={}
		d['log_len']=0
		for code in codes:
			d['code_'+str(code)]=0
			d['code_per_'+str(code)]=0
		for t in types:
			d['type_'+str(t)]=0
			d['type_per_'+str(t)]=0

		index=0
		Idx_d=copy.deepcopy(d)
		Idx_dict={}

		Idx=0
		for i in range(len_train+len_test):
			Idx=reader['Idx'][i]
			code=reader['LogInfo1'][i]
			t=reader['LogInfo2'][i]
			if Idx==Idxs[index]:
				Idx_d['log_len']+=1
				Idx_d['code_'+str(code)]+=1
				Idx_d['type_'+str(t)]+=1
			else:
				for code in codes:
					Idx_d['code_per_'+str(code)]=float(Idx_d['code_'+str(code)])/float(Idx_d['log_len'])
				for t in types:
					Idx_d['type_per_'+str(t)]=float(Idx_d['type_'+str(t)])/float(Idx_d['log_len'])

				Idx_dict[str(Idxs[index])]=self._row_info(Idx_d,codes,types)
				Idx_d=copy.deepcopy(d)
				index+=1

				Idx_d['log_len']+=1
				Idx_d['code_'+str(code)]+=1
				Idx_d['type_'+str(t)]+=1

		for code in codes:
			Idx_d['code_per_'+str(code)]=float(Idx_d['code_'+str(code)])/float(Idx_d['log_len'])
		for t in types:
			Idx_d['type_per_'+str(t)]=float(Idx_d['type_'+str(t)])/float(Idx_d['log_len'])

		Idx_dict[str(Idxs[index])]=self._row_info(Idx_d,codes,types)
		#self.output_info(Idx_dict)
		return Idx_dict

	def load_info2(self):
		"""
		距离交易前1周
		"""
		reader_train=pd.read_csv(self.config.path_origin_train_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		reader_test=pd.read_csv(self.config.path_origin_predict_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		len_train=len(reader_train)
		len_test=len(reader_test)
		reader=pd.concat([reader_train,reader_test],ignore_index=True)
		reader['Listinginfo1']=reader['Listinginfo1'].apply(self._deal_date)
		reader['LogInfo3']=reader['LogInfo3'].apply(self._deal_date)
		Idxs=self.get_Idx(reader['Idx'])
		codes=list(set(reader['LogInfo1']))
		types=list(set(reader['LogInfo2']))

		d={}
		d['log_len']=0
		for code in codes:
			d['code_'+str(code)]=0
			d['code_per_'+str(code)]=0
		for t in types:
			d['type_'+str(t)]=0
			d['type_per_'+str(t)]=0

		index=0
		Idx_d=copy.deepcopy(d)
		Idx_dict={}

		Idx=0
		for i in range(len_train+len_test):
			Idx=reader['Idx'][i]
			code=reader['LogInfo1'][i]
			t=reader['LogInfo2'][i]
			trading_date=reader['Listinginfo1'][i]
			log_date=reader['LogInfo3'][i]
			

			if Idx==Idxs[index]:
				if (trading_date-log_date)/86400<=7:
					Idx_d['log_len']+=1
					Idx_d['code_'+str(code)]+=1
					Idx_d['type_'+str(t)]+=1
			else:
				for code in codes:
					try:
						Idx_d['code_per_'+str(code)]=float(Idx_d['code_'+str(code)])/float(Idx_d['log_len'])
					except:
						Idx_d['code_per_'+str(code)]=0.0
				for t2 in types:
					try:
						Idx_d['type_per_'+str(t2)]=float(Idx_d['type_'+str(t2)])/float(Idx_d['log_len'])
					except:
						Idx_d['type_per_'+str(t2)]=0.0

				Idx_dict[str(Idxs[index])]=self._row_info(Idx_d,codes,types)
				Idx_d=copy.deepcopy(d)
				index+=1

				Idx_d['log_len']+=1
				Idx_d['code_'+str(code)]+=1
				Idx_d['type_'+str(t)]+=1

		for code in codes:
			try:
				Idx_d['code_per_'+str(code)]=float(Idx_d['code_'+str(code)])/float(Idx_d['log_len'])
			except:
				Idx_d['code_per_'+str(code)]=0.0
		for t2 in types:
			try:
				Idx_d['type_per_'+str(t2)]=float(Idx_d['type_'+str(t2)])/float(Idx_d['log_len'])
			except:
				Idx_d['type_per_'+str(t2)]=0.0

		Idx_dict[str(Idxs[index])]=self._row_info(Idx_d,codes,types)
		#self.output_info(Idx_dict)
		return Idx_dict

	def load_info3(self):
		"""
		时间处理
		"""
		reader_train=pd.read_csv(self.config.path_origin_train_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		reader_test=pd.read_csv(self.config.path_origin_predict_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		len_train=len(reader_train)
		len_test=len(reader_test)
		reader=pd.concat([reader_train,reader_test],ignore_index=True)
		reader['Listinginfo1']=reader['Listinginfo1'].apply(self._deal_date)
		reader['LogInfo3']=reader['LogInfo3'].apply(self._deal_date)
		Idxs=self.get_Idx(reader['Idx'])
		codes=list(set(reader['LogInfo1']))
		types=list(set(reader['LogInfo2']))

		index=0
		time_list=[]
		last_trading_time=0
		Idx=0
		X=[]
		Idx_dict={}
		for i in range(len_train+len_test):
			Idx=reader['Idx'][i]
			code=reader['LogInfo1'][i]
			t=reader['LogInfo2'][i]
			trading_date=reader['Listinginfo1'][i]
			log_date=reader['LogInfo3'][i]
			
			if Idx==Idxs[index]:
				time_list.append(log_date)
			else:
				Idx_dict[str(Idxs[index])]=self._deal_time_list(time_list,last_trading_time)
				time_list=[]

				time_list.append(log_date)
				index+=1
			last_trading_time=trading_date

		Idx_dict[str(Idxs[index])]=self._deal_time_list(time_list,last_trading_time)
		return Idx_dict


	def _deal_time_list(self,time_list,trading_date):
		time_list=sorted(time_list,reverse=True)
		log_len=len(time_list) #log长度
		first_log=trading_date-time_list[log_len-1] #交易时间-最开始log时间
		last_log=trading_date-time_list[0] #交易时间-最后log时间
		trade_date_log=0 #交易当天
		trade_date_per=0 
		last_date_log=0 #最后一天操作次数
		last_date_per=0
		last_3date_log=0 #最后三天操作次数
		last_3date_per=0
		last_5date_log=0 #最后5天操作次数
		last_5date_per=0
		last_7date_log=0 #最后7天操作次数
		last_7date_per=0
		active_dates=len(set(time_list)) #共活跃天数
		last_active_dates=0 #最后一周活跃天数
		last_active_per=0
		last_active_dates_list=[]
		for t in time_list:
			if t==trading_date:
				trade_date_log+=1
			if t==time_list[0]:
				last_date_log+=1
			if (time_list[0]-t)/86400<=3:
				last_3date_log+=1
			if (time_list[0]-t)/86400<=5:
				last_5date_log+=1
			if (time_list[0]-t)/86400<=7:
				last_7date_log+=1
				last_active_dates_list.append(t)

		trade_date_per=float(trade_date_log)/float(log_len)
		last_date_per=float(last_date_log)/float(log_len)
		last_3date_per=float(last_3date_log)/float(log_len)
		last_5date_per=float(last_5date_log)/float(log_len)
		last_7date_per=float(last_7date_log)/float(log_len)
		last_active_dates=len(set(last_active_dates_list))
		last_active_per=last_active_dates/active_dates

		l=[first_log,last_log,trade_date_log,trade_date_per,last_date_log,last_date_per,last_3date_log,last_3date_per,last_5date_log,last_5date_per,last_7date_log,last_7date_per,active_dates,last_active_dates,last_active_per]
		return l
		pass

	def _row_info(self,d,codes,types):
		l=[]
		l.append(d['log_len'])
		for code in codes:
			l.append(d['code_'+str(code)])
			l.append(d['code_per_'+str(code)])
		for t in types:
			l.append(d['type_'+str(t)])
			l.append(d['type_per_'+str(t)])
		return l

	def output_info(self):
		origin_instance=Load_origin_data(self.config)
		train_uids=origin_instance.load_train_uid()
		test_uids=origin_instance.load_predict_uid()
		Idx_dict=self.load_info()
		f1=open(self.config.path+"train/master_loginfo1.csv",'wb')
		f2=open(self.config.path+"test/master_loginfo1.csv",'wb')
		for uid in train_uids:
			if str(uid) in Idx_dict:
				l=Idx_dict[str(uid)]
			else:
				l=[0 for i in range(105)]
			f1.write(str(uid))
			for v in l:
				f1.write(','+str(v))
			f1.write('\n')

		for uid in test_uids:
			if str(uid) in Idx_dict:
				l=Idx_dict[str(uid)]
			else:
				l=[0 for i in range(105)]
			f2.write(str(uid))
			for v in l:
				f2.write(','+str(v))
			f2.write('\n')

		f1.close()
		f2.close()

	def output_info2(self):
		origin_instance=Load_origin_data(self.config)
		train_uids=origin_instance.load_train_uid()
		test_uids=origin_instance.load_predict_uid()
		Idx_dict=self.load_info2()
		f1=open(self.config.path+"train/master_loginfo_limit7.csv",'wb')
		f2=open(self.config.path+"test/master_loginfo_limit7.csv",'wb')
		for uid in train_uids:
			if str(uid) in Idx_dict:
				l=Idx_dict[str(uid)]
			else:
				l=[0 for i in range(105)]
			f1.write(str(uid))
			for v in l:
				f1.write(','+str(v))
			f1.write('\n')

		for uid in test_uids:
			if str(uid) in Idx_dict:
				l=Idx_dict[str(uid)]
			else:
				l=[0 for i in range(105)]
			f2.write(str(uid))
			for v in l:
				f2.write(','+str(v))
			f2.write('\n')

		f1.close()
		f2.close()

	def output_info3(self):
		origin_instance=Load_origin_data(self.config)
		train_uids=origin_instance.load_train_uid()
		test_uids=origin_instance.load_predict_uid()
		Idx_dict=self.load_info3()
		f1=open(self.config.path+"train/master_loginfo3.csv",'wb')
		f2=open(self.config.path+"test/master_loginfo3.csv",'wb')
		for uid in train_uids:
			if str(uid) in Idx_dict:
				l=Idx_dict[str(uid)]
			else:
				l=[0 for i in range(15)]
			f1.write(str(uid))
			for v in l:
				f1.write(','+str(v))
			f1.write('\n')

		for uid in test_uids:
			if str(uid) in Idx_dict:
				l=Idx_dict[str(uid)]
			else:
				l=[0 for i in range(15)]
			f2.write(str(uid))
			for v in l:
				f2.write(','+str(v))
			f2.write('\n')

		f1.close()
		f2.close()

	def get_Idx(self,col):
		last=-1000
		l=[]
		for v in col:
			if last!=v:
				l.append(v)
				last=v
		return l

	def _deal_date(self,n):
		t=time.strptime(str(n),"%Y-%m-%d")
		return time.mktime(t) #(time.mktime(t)-1262275200.0)/100


	def loginfo_idxs(self):
		reader_train=pd.read_csv(self.config.path_origin_train_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		reader_test=pd.read_csv(self.config.path_origin_predict_loginfo,iterator=False,delimiter=',',encoding='utf-8')
		len_train=len(reader_train)
		len_test=len(reader_test)
		reader=pd.concat([reader_train,reader_test],ignore_index=True)
		Idxs=self.get_Idx(reader['Idx'])
		return Idxs,len_train,len_test