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


class Merge(object):
	def __init__(self,config):
		self.config=config

	def load_file(self,level,name):
		reader=pd.read_csv(self.config.path_predict+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8')
		reader=np.array(reader)
		uid=reader[:,0].astype(int)
		scores=reader[:,1]
		scores=(scores-np.min(scores))/(np.max(scores)-np.min(scores))
		return uid,scores
	def output_file(self,level,name,test_uids,predicts):
		f1=open(self.config.path_predict+level+'/'+name+'.csv','wb')
		f1.write('"Idx","score"\n')
		for i in range(len(test_uids)):
			f1.write(str(test_uids[i])+","+str(predicts[i])+"\n")
		f1.close()


	def run(self):
		level='one'
		files=['xgb2000_20160330','xgb2000_20160330_2','xgb2000_20160330_3','xgb2000_20160330_4','xgb2000_20160330_5']
		test_uid=[]
		predicts=[]
		for f in files:
			uid,scores=self.load_file(level,f)
			if len(predicts)==0:
				test_uid=uid
				predicts=scores
			else:
				predicts+=scores
		
		predicts=predicts/len(files)
		self.output_file('one','merge',test_uid,predicts)


def main():
	m=Merge(Config())
	m.run()
	pass

if __name__ == '__main__':
	main()