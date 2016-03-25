#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

from config import Config

class Loginfo(object):
	def __init__(self,config):
		self.config=config
