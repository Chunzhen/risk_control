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
from datetime import datetime
import math

class Preprocessing(object):
	def __init__(self,config):
		self.config=config
		self.city_rank_list=[]

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
		reader_category=pd.concat([reader_train_category,reader_predict_category],ignore_index=True)
		reader_numeric=pd.concat([reader_train_numeric,reader_predict_numeric],ignore_index=True)
		return features_category,features_numeric,reader_category,reader_numeric,len_train,len_predict

	def dumps_scale(self):
		features_category,features_numeric,reader_category,reader_numeric,len_train,len_predict=self.load_data()
		i=0
		dumps=pd.DataFrame()
		category_num_set=set(['UserInfo_3','UserInfo_1','UserInfo_6','UserInfo_5','Education_Info1','Education_Info5','UserInfo_21','SocialNetwork_1','SocialNetwork_7','SocialNetwork_12','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15','UserInfo_16','UserInfo_17','SocialNetwork_2'])
		for feature in features_category:
			if feature=='UserInfo_24':
				continue
				reader_category[feature]=reader_category[feature].apply(self._deal_userInfo_24)
			elif feature=='UserInfo_2' or feature=='UserInfo_7' or feature=='UserInfo_4' or feature=='UserInfo_8' or feature=='UserInfo_20' or feature=='UserInfo_19': # or feature=='UserInfo_9'
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

		# dumps_numeric=pd.DataFrame()
		# i=0
		# for feature in features_numeric:	
		# 	if feature=='ListingInfo':
		# 		reader_numeric[feature]=reader_numeric[feature].apply(self._deal_date)
		# 	else:
		# 		reader_numeric[feature]=reader_numeric[feature].apply(self._deal_nan_digit)
		# 	col1=reader_numeric[feature].tolist()

		# 	col=set(col1)
		# 	if len(col)>100:		
		# 		nan_num=0
		# 		for v in col1:
		# 			if v==-1:
		# 				nan_num+=1
		# 		if nan_num<2000:
		# 			print 'add median'

		# 			self.col_median=np.median(col1)
		# 			print self.col_median
		# 			reader_numeric[feature]=reader_numeric[feature].apply(self._add_median)
		# 			# tmp_dict={}
		# 			# for vv in col1:
		# 			# 	nn=tmp_dict.get(str(vv),0)
		# 			# 	tmp_dict[str(vv)]=nn+1

		# 			# dd=sorted(tmp_dict.items(),key=lambda a:a[1],reverse=True)
		# 			# if dd[0][0]!='-1':
		# 			# 	self.col_median=dd[0][0]
		# 			# 	print dd[0][0]
		# 			# 	reader_numeric[feature]=reader_numeric[feature].apply(self._add_median)
			
		# 	#print len(col)
		# 	if len(col)<12:
		# 		tmp_dummys=pd.get_dummies(reader_numeric[feature])
		# 		if i==0:
		# 			dumps_numeric=tmp_dummys
		# 		else:
		# 			dumps_numeric=np.hstack((dumps_numeric,tmp_dummys))
		# 		i+=1

		# print dumps_numeric.shape
		#return 

		#X=np.hstack((reader_numeric,dumps))
		#X=np.hstack((X,dumps_numeric))

		#X=np.hstack((reader_numeric,dumps_numeric))
		#X=reader_numeric
		#X=np.hstack((reader_numeric,dumps_numeric))
		X=dumps
		print X.shape
		X_train=X[:len_train]
		X_predict=X[len_train:]
		one_value_col=[]

		tmp_X=[]
		for i in range(len(X_predict[0])):
			col_train_set=set(X_train[:,i])
			col_predict_set=set(X_predict[:,i])
			if len(col_train_set)==1: #reader_numeric[feature]
				one_value_col.append(i)
			else:
				tmp_X.append(X[:,i])

		X=np.array(tmp_X).transpose()
		X_train=X[:len_train]
		X_predict=X[len_train:]
		print X.shape
		return X_train,X_predict

	def _add_median(self,n):
		if n==-1:
			return self.col_median
		else:
			return n

	def category_num_scale(self):
		#category_num_set=['UserInfo_3','UserInfo_1','UserInfo_6','UserInfo_5','Education_Info1','Education_Info5','UserInfo_21','SocialNetwork_1','SocialNetwork_7','SocialNetwork_12','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15','UserInfo_16','UserInfo_17','SocialNetwork_2']
		category_num_set=['UserInfo_3','UserInfo_1','UserInfo_6','UserInfo_5','Education_Info1','Education_Info5','UserInfo_21','SocialNetwork_1','SocialNetwork_7','SocialNetwork_12','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15','UserInfo_16','UserInfo_17','SocialNetwork_2']
		tmp_set=['Education_Info2','Education_Info3','Education_Info4','Education_Info6','Education_Info7','Education_Info8','WeblogInfo_20','UserInfo_23','WeblogInfo_21','WeblogInfo_19','WeblogInfo_21','UserInfo_22']

		reader_category_train=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(category_num_set),encoding='utf-8')
		reader_category_predict=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(category_num_set),encoding='utf-8')
		reader1=pd.concat([reader_category_train,reader_category_predict],ignore_index=True)
		for feature in category_num_set:
			reader1[feature]=reader1[feature].apply(self._deal_nan_digit)


		reader_category_train2=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(tmp_set),encoding='utf-8')
		reader_category_predict2=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(tmp_set),encoding='utf-8')
		reader=pd.concat([reader_category_train2,reader_category_predict2],ignore_index=True)

		for feature in tmp_set:
			col=reader[feature]
			values=list(set(col))
			values=sorted(values)
			tmp_dict={}
			for i in range(len(values)):
				tmp_dict[str(values[i])]=i
			self.num_scale_dict=tmp_dict

			reader[feature]=reader[feature].apply(self.deal_num_scale)
		

		len_train=len(reader_category_train)
		len_predict=len(reader_category_predict)
		X=np.hstack((reader1,reader))
		X_train=X[:len_train]
		X_predict=X[len_train:]
		print X_train.shape
		print X_predict.shape
		return X_train,X_predict

	def deal_num_scale(self,n):
		#return self.num_scale_dict[str(n)]
		if self.num_scale_dict[str(n)]:
			return self.num_scale_dict[str(n)]
		else:
			#print self.num_scale_dict[str(n)]
			return -1

	def output_category_num_scale(self):
		X_train,X_predict=self.category_num_scale()
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_category_num2.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_category_num2.csv",seq=',',mode='wb',index=False,header=None)

	def output_dumps_scale(self):
		X_train,X_predict=self.dumps_scale()
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_category.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_category.csv",seq=',',mode='wb',index=False,header=None)

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

			if feature=='UserInfo_2' or feature=='UserInfo_4' or feature=='UserInfo_24' or feature=='UserInfo_20' or feature=='UserInfo_8':
				tmp_dummys=pd.get_dummies(reader_city)	
			else:
				tmp_dummys=pd.get_dummies(reader_province)
			#print tmp_dummys.shape
			#tmp_dummys2=pd.get_dummies(reader_city)
			
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
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_location3.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_location3.csv",seq=',',mode='wb',index=False,header=None)

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

	def _deal_longitude_scale(self,n):	
		try:
			location=self.location[n]
			return location[0]
		except:
			return 0# 100.223723

	def _deal_latitude_scale(self,n):
		try:
			location=self.location[n]
			return location[1]
		except:
			return 0# 34.480485

	def _deal_mean(self,n):
		if n==0:
			return self.col_mean
		else:
			return n

	def load_coor_json(self,feature):
		f=open(self.config.path_coor+feature+'.json')
		lines=f.readline()
		s=json.loads(u''+lines)
		return s

	def output_coor_scale(self):
		X_train,X_predict=self.coor_scale()
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_coor_mean.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_coor_mean.csv",seq=',',mode='wb',index=False,header=None)

	def coor_scale(self):
		features_category,features_numeric,reader_category,reader_numeric,len_train,len_predict=self.load_data()
		i=0
		dumps=[]
		for feature in features_category:
			if feature=='UserInfo_24' or feature=='UserInfo_2' or feature=='UserInfo_7' or feature=='UserInfo_4' or feature=='UserInfo_8' or feature=='UserInfo_20' or feature=='UserInfo_19':
				self.location=self.load_coor_json(feature)
				reader_province=reader_category[feature].apply(self._deal_longitude_scale)
				self.col_mean=np.mean(reader_province)
				reader_province=reader_province.apply(self._deal_mean)

				reader_city=reader_category[feature].apply(self._deal_latitude_scale)
				self.col_mean=np.mean(reader_city)
				reader_city=reader_city.apply(self._deal_mean)
			else:
				continue

			dumps.append(reader_province)
			dumps.append(reader_city)

		X=np.array(dumps).transpose()
		print X.shape
		X_train=X[:len_train]
		X_predict=X[len_train:]
		return X_train,X_predict


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
		reader_category=pd.concat([reader_train_category,reader_predict_category],ignore_index=True)
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

	
	def get_city_rank(self):
		f=open(self.config.path+'city_rank.csv')
		city_rank_list=[]
		for line in f.readlines():
			# l=line.split("、")
			# if len(l)==1:
			# 	l=line.split("，")
			# l=set(l)
			city_rank_list.append(u''+line)

		return city_rank_list

	def _deal_city_rank(self,s):
		try:
			location=self.location[s]
			#print location
			s=location[1].replace(u"市",'')
		except:
			return 0

		for i in range(len(self.city_rank_list)):
			tmp=self.city_rank_list[i]
			index=tmp.find(s)
			if index>0:
				return (5-i)
		return 0

	def city_rank_scale(self):
		category_num_set=['UserInfo_2','UserInfo_7','UserInfo_4','UserInfo_8','UserInfo_24','UserInfo_20','UserInfo_19']
		reader_category_train=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(category_num_set),encoding='utf-8')
		reader_category_predict=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(category_num_set),encoding='utf-8')
		len_train=len(reader_category_train)
		len_predict=len(reader_category_predict)

		reader=pd.concat([reader_category_train,reader_category_predict],ignore_index=True)
		self.city_rank_list=self.get_city_rank()

		for feature in category_num_set:
			self.location=self.load_location_json(feature)
			reader[feature]=reader[feature].apply(self._deal_city_rank)

		reader=np.array(reader)

		m,n=reader.shape
		cor_col=[]
		for i in range(n-1):
			col1=reader[:,i]
			#print col1
			#print col1.shape
			#break
			for j in range(i+1,n):
				col2=reader[:,j]
				col=col1-col2
				cor_col.append(col-np.min(col))

		cor_col=np.transpose(np.array(cor_col))
		X=np.hstack((reader,cor_col))
		X_train=X[:len_train]
		X_predict=X[len_train:]
		print X.shape
		return X_train,X_predict

	def output_city_rank(self):
		X_train,X_predict=self.city_rank_scale()
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_city_rank.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_city_rank.csv",seq=',',mode='wb',index=False,header=None)


	def missing_value_scale(self):
		"""
		缺失值当做一列特征
		"""
		features_category,features_numeric,reader_category,reader_numeric,len_train,len_predict=self.load_data()
		print reader_category.shape
		print reader_numeric.shape

		for feature in features_category:
			if feature=='UserInfo_24' or feature=='Education_Info3' or feature=='Education_Info4':
				reader_category[feature]=reader_category[feature].apply(self.is_E)
			else:
				reader_category[feature]=reader_category[feature].apply(self.is_missing)

		for feature in features_numeric:
			reader_numeric[feature]=reader_numeric[feature].apply(self.is_missing)

		X=np.hstack((reader_category,reader_numeric))
		X_train=X[:len_train]
		X_predict=X[len_train:]
		print X.shape
		return X_train,X_predict

	def output_missing_scale(self):
		X_train,X_predict=self.missing_value_scale()
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_missing_scale.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_missing_scale.csv",seq=',',mode='wb',index=False,header=None)


	def is_E(self,n): #UserInfo_24 Education_Info3 Education_Info4 
		if str(n)=='E' or str(n)=='D':
			return 0
		else:
			return 1


	def is_missing(self,n):
		if str(n)=='nan' or str(n)=='-1' or str(n)==u'不详':
			return 0
		else:
			return 1

	def listinfo_transform(self):
		reader1=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(['ListingInfo']),encoding='utf-8')
		reader2=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(['ListingInfo']),encoding='utf-8')
		len_train=len(reader1)
		len_predict=len(reader2)
		reader=pd.concat([reader1,reader2],ignore_index=True)
		year=reader['ListingInfo'].apply(self._date_to_year)
		month=reader['ListingInfo'].apply(self._date_to_month)
		week=reader['ListingInfo'].apply(self._date_to_week)
		day=reader['ListingInfo'].apply(self._date_to_day)
		is_week_day=reader['ListingInfo'].apply(self._date_is_week_day)

		month_dumps=pd.get_dummies(month)
		week_dumps=pd.get_dummies(week)

		X=[year,month,week,day,is_week_day]
		X=np.array(X).transpose()
		X=np.hstack((X,month_dumps,week_dumps))
		print X.shape
		X_train=X[:len_train]
		X_predict=X[len_train:]
		pd.DataFrame(X_train).to_csv(self.config.path+"train/listingInfo_transform.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/listingInfo_transform.csv",seq=',',mode='wb',index=False,header=None)

	def _date_to_year(self,n):
		try:
			t=time.strptime(str(n),"%Y/%m/%d")
		except:
			t=time.strptime(str(n),"%d/%m/%Y")
		d=datetime.fromtimestamp(time.mktime(t))
		return d.year

	def _date_to_month(self,n):
		try:
			t=time.strptime(str(n),"%Y/%m/%d")
		except:
			t=time.strptime(str(n),"%d/%m/%Y")
		d=datetime.fromtimestamp(time.mktime(t))
		return d.month

	def _date_to_day(self,n):
		try:
			t=time.strptime(str(n),"%Y/%m/%d")
		except:
			t=time.strptime(str(n),"%d/%m/%Y")
		d=datetime.fromtimestamp(time.mktime(t))
		return d.day

	def _date_to_week(self,n):
		try:
			t=time.strptime(str(n),"%Y/%m/%d")
		except:
			t=time.strptime(str(n),"%d/%m/%Y")
		d=datetime.fromtimestamp(time.mktime(t))
		return d.weekday()+1
	def _date_is_week_day(self,n):
		try:
			t=time.strptime(str(n),"%Y/%m/%d")
		except:
			t=time.strptime(str(n),"%d/%m/%Y")
		d=datetime.fromtimestamp(time.mktime(t))
		if d.weekday()<5:
			return 1
		else:
			return 0


	def education_transform(self):
		f='category'
		origin_instance=Load_origin_data(self.config)
		#features=origin_instance.load_feature(f)
		#features=['UserInfo_3','UserInfo_1','UserInfo_6','UserInfo_5','UserInfo_9','UserInfo_21','UserInfo_23','UserInfo_22','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15','UserInfo_16','UserInfo_17']
		#features=['UserInfo_6','UserInfo_5','UserInfo_9','UserInfo_21','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_16','UserInfo_17']
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
		reader2=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(features),encoding='utf-8')
		len_train=len(reader1)
		len_predict=len(reader2)
		reader=pd.concat([reader1,reader2],ignore_index=True)
		self.sample_num=len_train+len_predict
		
		y=origin_instance.load_train_y()

		l=[]
		for feature in features:
			s=list(set(reader[feature]))
			feature_d={}
			for v in s:
				feature_d[str(v)]=0
			self.feature_d=feature_d
			reader[feature].apply(self._count_feature_num)
			#print self.feature_d
			tmp_l=reader[feature].apply(self._count_feature_per)
			#print tmp_l

			cor=np.corrcoef(np.array(tmp_l)[:len_train],y)[0,1]
			print cor

			l.append(tmp_l)
		
		X=np.array(l).transpose()
		print X.shape
		X_train=X[:len_train]
		X_predict=X[len_train:]
		pd.DataFrame(X_train).to_csv(self.config.path+"train/master_"+f+"_weight.csv",seq=',',mode='wb',index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"test/master_"+f+"_weight.csv",seq=',',mode='wb',index=False,header=None)

	def _count_feature_num(self,n):
		self.feature_d[str(n)]+=1
	def _count_feature_per(self,n):
		return float(self.feature_d[str(n)])/float(self.sample_num)


def main():
	instance=Preprocessing(Config())
	instance.education_transform()
	pass

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')
	main()





	



