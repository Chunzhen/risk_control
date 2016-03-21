#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config
from load_origin_data import Load_origin_data

class Preprocessing(object):
	def __init__(self,config):
		self.config=config

	def load_data(self):
		"""
		特征处理
		"""
		pass

	def scale_func(self):
		"""
		处理函数
		"""
		pass

	def output_data(self):
		"""
		输出函数
		"""
		pass

	def run(self):
		"""
		实例函数
		"""
		pass

	



