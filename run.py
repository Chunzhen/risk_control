	#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os

from config import Config
from load_origin_data import Load_origin_data
from analysis import Analysis
from preprocessing import Preprocessing
from load_scale_data import Load_scale_data
from mboost import Mboost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Lasso

from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import RandomizedLogisticRegression

class Run(object):
	def __init__(self):
		self.config=Config()
		self.params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':12.56, #27802.0/(2198.0) 
	    'eval_metric': 'auc',
	    'gamma':0.05,
	    'max_depth':6,#11
	    'lambda':100,
	    'subsample':0.7,
	    'colsample_bytree':0.5,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':7,
	    'nthread':4,
	    'silent':1
	    }

	def level_one_wrapper(self):
		# data_instance=Load_scale_data(self.config)
		# origin_instance=Load_origin_data(self.config)
		# X=data_instance.load_train_X()
		# y=origin_instance.load_train_y()
		# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
		# clf=LogisticRegression(max_iter=1000,C=0.0005,class_weight={1:2198,0:27802},solver='lbfgs',tol=0.001)
		# clf.fit(X_train,y_train)
		# y_pred=clf.predict_proba(X_test)
		# y_pred=y_pred[:,1]
		# auc_score=metrics.roc_auc_score(y_test,y_pred)
		# print auc_score
		# return

		level='one'
		data_instance=Load_scale_data(self.config)
		X_0,X_1,uid_0,uid_1=data_instance.load_train_X_separate()
		
		mboost_instance=Mboost(self.config)
		mboost_instance.xgb_level_train(level,'xgb1000_master_ThirdParty_same_period_selection',X_0,X_1,uid_0,uid_1,self.params,800)
		#mboost_instance.level_train(AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=20,max_depth=6,min_samples_split=9),n_estimators=100,learning_rate=0.02),level,'ada100',X_0,X_1,uid_0,uid_1)
		#mboost_instance.level_train(RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_split=9),level,'rf100',X_0,X_1,uid_0,uid_1)
		#mboost_instance.level_train(LogisticRegression(max_iter=1500,C=0.0005,class_weight={0:2198,1:27802},solver='lbfgs',tol=0.001),level,'lr_x',X_0,X_1,uid_0,uid_1)
		#mboost_instance.level_train(GradientBoostingClassifier(n_estimators=20,max_depth=11,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,'gbdt20',X_0,X_1,uid_0,uid_1)
		#max_iter=1000,alpha=0.0005,max_iter=1000,,class_weight={0:2198,1:27802},solver='auto'
	def level_one_predict(self):
		level='one'
		origin_instance=Load_origin_data(self.config)
		data_instance=Load_scale_data(self.config)
		X_0,X_1,uid_0,uid_1=data_instance.load_train_X_separate()

		#predict
		predict_X=data_instance.load_predict_X()
		predict_uid=origin_instance.load_predict_uid()

		mboost_instance=Mboost(self.config)
		
		mboost_instance.xgb_predict(level, 'xgb900_master_467_third_party', X_0, X_1, predict_X, predict_uid, self.params, 900)
		#mboost_instance.level_predict(LogisticRegression(max_iter=1500,C=0.0005,class_weight={0:2198,1:27802},tol=0.001),level,'lr2',X_0,X_1, predict_X, predict_uid)

def main():
	instance=Run()
	instance.level_one_wrapper()
	#instance.level_one_predict()

if __name__ == '__main__':
	main()
