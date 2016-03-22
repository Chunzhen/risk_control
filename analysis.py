#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False


class Analysis(object):
	def __init__(self,config):
		self.config=config

	def features_scale(self,ftype):
		instance=Load_origin_data(self.config)
		reader_category=pd.DataFrame()
		reader_numeric=pd.DataFrame()
		features_category,features_numeric=instance.load_feature_type()
		if ftype=='train':
			reader_category,reader_numeric=instance.load_train_X()
		else:
			reader_category,reader_numeric=instance.load_predict_X()

		scales_category={}
		for feature in features_category:
			scale=self._get_features_scale(reader_category[feature])
			scales_category[feature]=scale
		scales_numeric={}
		for feature in features_numeric:
			scale=self._get_features_scale(reader_numeric[feature])
			scales_numeric[feature]=scale

		return scales_category,scales_numeric

	def _get_features_scale(self,col):
		d={}
		for val in col:
			val=str(val)
			num=d.get(val,0)
			d[val]=num+1
		d=sorted(d.items(),key=lambda a:a[0],reverse=False)
		return d

	def output_features_scale(self,ftype):
		f=None
		if ftype=='train':
			f1=open(self.config.path_analysis+'features_scale_train_category.csv','wb')
			f2=open(self.config.path_analysis+'features_scale_train_numeric.csv','wb')
		else:
			f1=open(self.config.path_analysis+'features_scale_predict_category.csv','wb')
			f2=open(self.config.path_analysis+'features_scale_predict_numeric.csv','wb')

		scales_category,scales_numeric=self.features_scale(ftype)

		for feature,scale in scales_category.items():
			f1.write(feature)
			for val,num in scale:
				f1.write(', ['+str(val)+','+str(num)+']')
			f1.write('\n')

		for feature,scale in scales_numeric.items():
			f2.write(feature)
			for val,num in scale:
				f2.write(', ['+str(val)+','+str(num)+']')
			f2.write('\n')

		f1.close()
		f2.close()

	def print_features_scale(self,ftype):
		scales_category,scales_numeric=self.features_scale(ftype)
		i=0
		# for feature,scale in scales_category.items():
		# 	x=[]
		# 	y=[]
		# 	for val,num in scale:
		# 		x.append(val)
		# 		y.append(num)
		# 	self.plot_one_x_num(x,y, feature, ftype, 'category')

		for feature,scale in scales_numeric.items():
			x=[]
			y=[]
			i=0
			for val,num in scale:
				if val=='nan':
					val=-1
				try:
					x.append(float(val))
				except:
					x.append(i)
				i+=1
				y.append(num)
			self.plot_one_x_num(x,y, feature, ftype, 'numeric')

	def plot_one_x_num(self,x1,y1,title,ftype,feature_type):
		x=[]
		if feature_type=='numeric':
			x=x1
		else:
			x=[i for i in range(len(x1))]

		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.scatter(x,y1,color="red",alpha=0.4)
		#plt.xticks(x,x1,rotation='vertical')	
		#ax.scatter(x2,y2,color="green",alpha=0.4)
			
		plt.title("feature: "+title)

		#plt.grid(True)
		plt.show()
		self.savefig(self.config.path_analysis+'plot_'+ftype+'/'+feature_type+'/'+title+'.png')
		pass

	def features_scale_separate(self):
		instance=Load_origin_data(self.config)
		reader_category=pd.DataFrame()
		reader_numeric=pd.DataFrame()
		features_category,features_numeric=instance.load_feature_type()
		
		reader_one_category,reader_one_numeric,reader_zero_category,reader_zero_numeric=instance.load_train_X_separate()

		scales_one_category={}
		for feature in features_category:
			scale=self._get_features_scale(reader_one_category[feature])
			scales_one_category[feature]=scale
		scales_one_numeric={}
		for feature in features_numeric:
			scale=self._get_features_scale(reader_one_numeric[feature])
			scales_one_numeric[feature]=scale

		scales_zero_category={}
		for feature in features_category:
			scale=self._get_features_scale(reader_zero_category[feature])
			scales_zero_category[feature]=scale
		scales_zero_numeric={}
		for feature in features_numeric:
			scale=self._get_features_scale(reader_zero_numeric[feature])
			scales_zero_numeric[feature]=scale

		return scales_one_category,scales_one_numeric,scales_zero_category,scales_zero_numeric

	def print_features_scale_separate(self):
		scales_one_category,scales_one_numeric,scales_zero_category,scales_zero_numeric=self.features_scale_separate()
		i=0

		for feature,scale in scales_one_category.items():
			x1=[]
			y1=[]
			x2=[]
			y2=[]
			zero_scale=scales_zero_category[feature]
			d={}
			i=0
			for val,num in scale:
				index=d.get(val,-1)
				if index==-1:
					d[val]=i
					i+=1
			for val,num in zero_scale:
				index=d.get(val,-1)
				if index==-1:
					d[val]=i
					i+=1
			for val,num in scale:
				x1.append(d[val])
				y1.append(-num-10)
			for val,num in zero_scale:
				x2.append(d[val])
				y2.append(num+10)

			self.plot_two_x_num(x1,y1,x2,y2,feature, 'category')
			#break

		for feature,scale in scales_one_numeric.items():
			x1=[]
			y1=[]
			x2=[]
			y2=[]
			i=0
			zero_scale=scales_zero_numeric[feature]
			for val,num in scale:
				if val=='nan':
					val=-1
				try:
					x1.append(float(val))
				except:
					x1.append(i)
				i+=1
				y1.append(-num-10)

			i=0
			for val,num in zero_scale:
				if val=='nan':
					val=-1
				try:
					x2.append(float(val))
				except:
					x2.append(i)
				i+=1
				y2.append(num+10)
			self.plot_two_x_num(x1,y1,x2,y2, feature, 'numeric')

	def plot_two_x_num(self,x1,y1,x2,y2,title,feature_type):

		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.scatter(x1,y1,color="red",alpha=0.4)
		#plt.xticks(x,x1,rotation='vertical')	
		ax.scatter(x2,y2,color="green",alpha=0.4)
			
		plt.title("feature: "+title)

		#plt.grid(True)
		plt.show()
		self.savefig(self.config.path_analysis+'plot_separate/'+feature_type+'/'+title+'.png')
		pass

	def savefig(self,path):
		plt.savefig(path)











