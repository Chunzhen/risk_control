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
from sklearn.ensemble import BaggingClassifier

class Run(object):
	def __init__(self):
		self.config=Config()

	def level_one_wrapper(self):
		level='one'
		data_instance=Load_scale_data(self.config)
		X_0,X_1,uid_0,uid_1=data_instance.load_train_X_separate()
		params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':27802.0/(2198.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':11,#11
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':7,
	    'nthread':4
	    }
		mboost_instance=Mboost(self.config)
		#mboost_instance.xgb_level_train(level,'xgb2000_dumps_location',X_0,X_1,uid_0,uid_1,params,2000)
		#mboost_instance.level_train(RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_split=20),level,'rf100',X_0,X_1,uid_0,uid_1)
		#mboost_instance.level_train(LogisticRegression(),level,'lr_sag',X_0,X_1,uid_0,uid_1)
		mboost_instance.level_train(GradientBoostingClassifier(n_estimators=20,max_depth=11,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,'gbdt20',X_0,X_1,uid_0,uid_1)
		

	def level_one_predict(self):
		level='one'
		origin_instance=Load_origin_data(self.config)
		data_instance=Load_scale_data(self.config)
		X_0,X_1,uid_0,uid_1=data_instance.load_train_X_separate()

		#predict
		predict_X=data_instance.load_predict_X()
		predict_uid=origin_instance.load_predict_uid()

		params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':27802.0/(2198.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':11,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':7,
	    'nthread':4
	    }
		mboost_instance=Mboost(self.config)
		mboost_instance.xgb_predict(level, 'xgb3000', X_0, X_1, predict_X, predict_uid, params, 3000)

def main():
	instance=Run()
	instance.level_one_wrapper()
	#instance.level_one_predict()

if __name__ == '__main__':
	main()
