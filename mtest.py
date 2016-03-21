#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os

from config import Config
from load_origin_data import Load_origin_data

class Mtest(object):
	def __init__(self):
		self.config=Config()
		pass
	def load_feature_type_test(self):
		instance=Load_origin_data(self.config)
		features_category,features_numeric=instance.load_feature_type()
		print len(features_category),' ',len(features_numeric)
		print features_numeric
		pass

	def load_train_X_test(self):
		instance=Load_origin_data(self.config)
		reader_category,reader_numeric=instance.load_train_X()
		print reader_category.shape
		print reader_numeric.shape

	def load_predict_X_test(self):
		instance=Load_origin_data(self.config)
		reader_category,reader_numeric=instance.load_predict_X()
		print reader_category.shape
		print reader_numeric.shape

	def load_train_y_test(self):
		instance=Load_origin_data(self.config)
		y=instance.load_train_y()
		print y
		print len(y)

	def load_train_uid_test(self):
		instance=Load_origin_data(self.config)
		uid=instance.load_train_uid()
		print len(uid)

	def load_predict_uid_test(self):
		instance=Load_origin_data(self.config)
		uid=instance.load_predict_uid()
		print len(uid)

	def run(self,n):
		if n==0:
			self.load_feature_type_test()
		elif n==1:
			self.load_train_X_test()
		elif n==2:
			self.load_predict_X_test()
		elif n==3:
			self.load_train_y_test()
		elif n==4:
			self.load_train_uid_test()
		elif n==5:
			self.load_predict_uid_test()

def main():
	test_instance=Mtest()
	test_instance.run(5)
	pass

if __name__ == '__main__':
	main()