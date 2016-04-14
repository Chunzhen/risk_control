#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

import time
import copy

from config import Config
from load_origin_data import Load_origin_data

from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

from mboost_thread import Mboost_thread
from xgb_mboost_thread import Xgb_mboost_thread

class Mboost(object):
	"""
	:class Mboost
	:独立出训练与测试的类，多层模型时可调用相同类，达到代码重用
	"""
	def __init__(self,config):
		"""
		:type config: Config
		:初始化配置信息
		"""
		self.config=config
		pass

	def fold(self,len_0,len_1,n_folds):
		"""
		:type len_0: int
		:type len_1: int
		:type n_folds: int
		:rtype f0: List[List[int]]
		:rtype f1: List[List[int]]
		:将正类与负类分别分为n折，返回这n折每一折的下标集合
		"""
		random_state=self.config.fold_random_state #设置随机值
		kf0=KFold(n=len_0, n_folds=n_folds, shuffle=True,random_state=random_state)
		kf1=KFold(n=len_1,n_folds=n_folds, shuffle=True,random_state=random_state)
		f0=[]
		f1=[]
		#分装k fold的下标并返回
		for train_index_0,test_index_0 in kf0:
			f0.append([train_index_0.tolist(),test_index_0.tolist()])
		for train_index_1,test_index_1 in kf1:
			f1.append([train_index_1.tolist(),test_index_1.tolist()])
		return f0,f1


	def level_train(self,clf,level,name,X_0,X_1,uid_0,uid_1):
		"""
		:type clf: scikit-learn classifier or regressor scikit-learn分类器或回归器
		:type level: str 训练第几层
		:type name: str 分类器命名
		:type X_0: numpy.array 类别0特征矩阵
		:type X_1: numpy.array 类型1特征矩阵
		:type uid_0: List 类别0 uid
		:type uid_1: List 类别1 uid
		:层次训练方法，将正负类数据分别分为n folds，然后用(n-1) folds作为训练集
		:用1 fold作为测试集，循环训练，得到一维特征的数据输出
		:每次训练n folds个模型
		"""
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds) #获得n folds 下标

		predicts=[]
		test_uids=[]
		scores=[]
		part_uids=[]

		threads=[]

		for i in range(n_folds):
			train_index_0,test_index_0=f0[i][0],f0[i][1]
			train_index_1,test_index_1=f1[i][0],f1[i][1]

			train_1=X_1[train_index_1]
			test_1=X_1[test_index_1]

			train_0=X_0[train_index_0]
			test_0=X_0[test_index_0]

			test_uid_1=uid_1[test_index_1]
			test_uid_0=uid_0[test_index_0]

			#类的标签直接获得
			y_train=np.hstack((np.ones(len(train_1)),np.zeros(len(train_0))))
			y_test=np.hstack((np.ones(len(test_1)),np.zeros(len(test_0))))

			#测试fold中的uid合并
			test_uid=np.hstack((test_uid_1,test_uid_0))

			#训练集和测试集
			x_train=np.vstack((train_1,train_0))
			x_test=np.vstack((test_1,test_0))

			threads.append(Mboost_thread(clf,x_train, y_train, x_test,y_test,test_uid))

		for thread in threads:
			thread._run()

		# for thread in threads:
		# 	thread.join()
		

		for thread in threads:
			auc_score=thread.auc_score
			predicts.extend(thread.predict)
			test_uids.extend(thread.test_uid)
			scores.append(auc_score)
			print auc_score		

		#保存输出结果
		d={}
		#print test_uids
		for k, uid in enumerate(test_uids):
			d[str(uid)]=predicts[k]
		origin_instance=Load_origin_data(self.config)

		uids=origin_instance.load_train_uid()

		#print uids
		predicts2=[]
		for uid in uids:
			predicts2.append(d[str(uid)])
		
		self.output_level_train(predicts2,uids,scores,level,name)
		print name+" average scores:",np.mean(scores)

	def fpreproc(self,dtrain, dtest, param):
		label = dtrain.get_label()
		ratio = float(np.sum(label == 0)) / np.sum(label==1)
		param['scale_pos_weight'] = ratio
		return (dtrain, dtest, param)

	def logregobj(preds, dtrain):
		labels = dtrain.get_label()
		preds = 1.0 / (1.0 + np.exp(-preds))
		grad = preds - labels
		hess = preds * (1.0-preds)
		return grad, hess

	def xgb_level_train(self,level,name,X_0,X_1,uid_0,uid_1,params,round):
		"""
		:type level: str 训练第几层
		:type name: str 分类器命名
		:type X_0: numpy.array 类别0特征矩阵
		:type X_1: numpy.array 类型1特征矩阵
		:type uid_0: List 类别0 uid
		:type uid_1: List 类别1 uid
		:type params: dict XGBoost的配置参数
		:type round: int XGBoost的迭代次数
		:与level train功能一致，只是分类器调用XGBoost实现的分类器
		"""
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds)

		predicts=[]
		test_uids=[]
		scores=[]
		part_uids=[]

		x_train=np.vstack((X_0,X_1))
		y_train=np.hstack((np.zeros(len(X_0)),np.ones(len(X_1))))
		dtrain=xgb.DMatrix(x_train,label=y_train)

		log=xgb.cv(params,dtrain,round,nfold=5,metrics={'auc'},seed=1,verbose_eval=10)#,fpreproc=self.fpreproc ,obj=logregobj
		log.to_csv(self.config.path_train+level+'/'+name+'.log',seq=',',mode='wb',index=False)

		# model=xgb.train(params,dtrain,num_boost_round=round,verbose_eval=10,evals=[(dtrain,'train')])
		# feature_score=model.get_fscore()
		# sorted_features=sorted(feature_score.items(),key=lambda d:d[1],reverse=True)
		# f=open(self.config.path+'train/features_importance.csv','wb')
		# for feature,score in sorted_features:
		# 	f.write(feature[1:]+','+str(score)+'\n')
		# return 

		threads=[]

		for i in range(n_folds):
			train_index_0,test_index_0=f0[i][0],f0[i][1]
			train_index_1,test_index_1=f1[i][0],f1[i][1]

			train_1=X_1[train_index_1]
			test_1=X_1[test_index_1]

			train_0=X_0[train_index_0]
			test_0=X_0[test_index_0]

			test_uid_1=uid_1[test_index_1]
			test_uid_0=uid_0[test_index_0]

			#train_1=np.vstack((train_1,train_1))

			y_train=np.hstack((np.ones(len(train_1)),np.zeros(len(train_0))))		
			y_test=np.hstack((np.ones(len(test_1)),np.zeros(len(test_0))))

			test_uid=np.hstack((test_uid_1,test_uid_0))

			x_train=np.vstack((train_1,train_0))
			x_test=np.vstack((test_1,test_0))

			dtest=xgb.DMatrix(x_test)
			dval=xgb.DMatrix(x_test,label=y_test)
			dtrain=xgb.DMatrix(x_train,label=y_train)
			watchlist=[(dval,'val'),(dtrain,'train')]

			#xgb.cv(params,dtrain,round,nfold=5,metrics={'auc'},seed=7,show_progress=10)
			threads.append(Xgb_mboost_thread(dtrain,y_train,dtest,y_test,test_uid,watchlist,params,round))

		for thread in threads:
			thread.start()

		for thread in threads:
			thread.join()

		for thread in threads:
			auc_score=thread.auc_score
			predicts.extend(thread.predict.tolist())
			test_uids.extend(thread.test_uid.tolist())
			scores.append(auc_score)
			print auc_score			

		#保存输出结果
		d={}
		#print test_uids
		for k, uid in enumerate(test_uids):
			d[str(uid)]=predicts[k]
		origin_instance=Load_origin_data(self.config)

		uids=origin_instance.load_train_uid()

		#print uids
		predicts2=[]
		for uid in uids:
			predicts2.append(d[str(uid)])
		
		self.output_level_train(predicts2,uids,scores,level,name)
		print name+" average scores:",np.mean(scores)

	def output_level_train(self,predicts,test_uids,scores,level,name):	
		"""
		:type predicts: List[float] 预测值列表
		:type test_uids: List[str] 预测uid
		:type scores: List[float] 每一折的AUC得分
		:type level: str 训练第几层
		:type name: str 分类器命名
		:输出每层每个分类器的预测结果到文件
		"""
		f1=open(self.config.path_train+level+'/'+name+'.csv','wb')
		f2=open(self.config.path_train+level+'/'+name+'_score.csv','wb')
		for i in range(len(test_uids)):
			f1.write(str(test_uids[i])+","+str(predicts[i])+"\n")

		for score in scores:
			f2.write(str(score)+"\n")

		f1.close()
		f2.close()

	def level_predict(self,clf,level,name,X_0,X_1,predict_X,predict_uid):
		"""
		:type clf: scikit-learn classifier or regressor scikit-learn分类器或回归器
		:type level: str 预测第几层
		:type name: str 分类器命名
		:type X_0: numpy.array 类别0特征矩阵
		:type X_1: numpy.array 类型1特征矩阵
		:type predict_X: 预测集的特征矩阵
		:type predict_uid: 预测集的uid
		:层次预测，每次只训练1个模型，预测1个结果
		"""
		start=datetime.now()
		x_train=np.vstack((X_1,X_0))
		y_train=np.hstack((np.ones(len(X_1)),np.zeros(len(X_0))))

		clf.fit(x_train,y_train)
		try:
			pred_result=clf.predict_proba(predict_X)
			self.output_level_predict(pred_result[:,1],predict_uid,level,name)
		except:
			pred_result=clf.predict(predict_X)
			self.output_level_predict(pred_result,predict_uid,level,name)
		
		end=datetime.now()
		print "finish predict:"+name+" Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"

	def xgb_predict(self,level,name,X_0,X_1,predict_X,predict_uid,params,round):
		"""
		:type name: str 分类器命名
		:type X_0: numpy.array 类别0特征矩阵
		:type X_1: numpy.array 类型1特征矩阵
		:type predict_X: 预测集的特征矩阵
		:type predict_uid: 预测集的uid
		:type params: dict XGBoost的配置参数
		:type round: int XGBoost的迭代次数
		:XGBoost预测，每次只训练1个模型，预测1个结果
		"""
		start=datetime.now()
		x_train=np.vstack((X_1,X_0))
		
		y_train=np.hstack((np.ones(len(X_1)),np.zeros(len(X_0))))
		
		dtrain=xgb.DMatrix(x_train,label=y_train)
		watchlist=[(dtrain,'train')]
		model=xgb.train(params,dtrain,num_boost_round=round,evals=watchlist,verbose_eval=100)

		dpredict=xgb.DMatrix(predict_X)
		predict_result=model.predict(dpredict)
		self.output_level_predict(predict_result,predict_uid,level,name)
		end=datetime.now()
		print "finish predict:"+name+" Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"

	def output_level_predict(self,predicts,test_uids,level,name):	
		"""
		:type predicts: List[float] 预测值列表
		:type test_uids: List[str] 预测uid
		:type level: str 训练第几层
		:type name: str 分类器命名
		:输出预测结果到文件
		"""
		f1=open(self.config.path_predict+level+'/'+name+'.csv','wb')
		f1.write('"Idx","score"\n')
		for i in range(len(test_uids)):
			f1.write(str(test_uids[i])+","+str(predicts[i])+"\n")
		f1.close()