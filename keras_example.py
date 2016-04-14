	#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
from config import Config
from load_origin_data import Load_origin_data
from analysis import Analysis
from preprocessing import Preprocessing
from load_scale_data import Load_scale_data
from sklearn.cross_validation import train_test_split

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn import metrics
from sklearn.metrics import roc_curve, auc

class Keras_example(object):

	def __init__(self,config):
		self.config=config

	def normalizer_scale(self,X):
		scaler=Normalizer()
		return scaler.fit_transform(X)

	def classifier(self):
		level='one'
		data_instance=Load_scale_data(self.config)
		X_0,X_1,uid_0,uid_1=data_instance.load_train_X_separate()
		X_0_train,X_0_test=train_test_split(X_0,test_size=0.2)
		X_1_train,X_1_test=train_test_split(X_1,test_size=0.2)
		
		X_train=np.vstack((X_1_train,X_0_train))
		X_test=np.vstack((X_1_test,X_0_test))

		y_train=np.hstack((np.ones(len(X_1_train)),np.zeros(len(X_0_train))))	
		y_test=np.hstack((np.ones(len(X_1_test)),np.zeros(len(X_0_test))))


		batch_size = 128
		nb_classes = 2
		nb_epoch = 10

		# the data, shuffled and split between train and test sets

		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')

		print(X_train.shape[0], 'train samples')
		print(X_test.shape[0], 'test samples')

		# convert class vectors to binary class matrices
		Y_train = np_utils.to_categorical(y_train, nb_classes)
		Y_test = np_utils.to_categorical(y_test, nb_classes)


		model = Sequential() 
		model.add(Dense(50,input_shape=(X_train.shape[1],)))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
		model.add(Dense(50)) 
		model.add(Activation('linear'))
		model.add(Dropout(0.2))
		model.add(Dense(2))
		model.add(Activation('sigmoid'))

		#model.summary()
		sgd = SGD(lr=0.001, decay=1e-6, momentum=0.01, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

		history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

		score = model.evaluate(X_test, Y_test, verbose=0)
		y_pred=model.predict_proba(X_test)
		print y_pred
		print len(y_pred[:,0]),sum(y_pred[:,0])

		auc_score=metrics.roc_auc_score(y_test,y_pred[:,0])
		print auc_score
		print('Test score:', score[0])
		print('Test accuracy:', score[1])

		# batch_size = 128
		# nb_classes = 10
		# nb_epoch = 5

		# # the data, shuffled and split between train and test sets
		# (X_train, y_train), (X_test, y_test) = mnist.load_data()

		# X_train = X_train.reshape(60000, 784)
		# X_test = X_test.reshape(10000, 784)
		# X_train = X_train.astype('float32')
		# X_test = X_test.astype('float32')
		# X_train /= 255
		# X_test /= 255
		# print(X_train.shape[0], 'train samples')
		# print(X_test.shape[0], 'test samples')

		# # convert class vectors to binary class matrices
		# Y_train = np_utils.to_categorical(y_train, nb_classes)
		# Y_test = np_utils.to_categorical(y_test, nb_classes)

		# model = Sequential()
		# model.add(Dense(512, input_shape=(784,)))
		# model.add(Activation('relu'))
		# model.add(Dropout(0.2))
		# model.add(Dense(512))
		# model.add(Activation('relu'))
		# model.add(Dropout(0.2))
		# model.add(Dense(10))
		# model.add(Activation('softmax'))

		# #model.summary()

		# model.compile(loss='categorical_crossentropy',
		#               optimizer=RMSprop(),
		#               metrics=['accuracy'])

		# history = model.fit(X_train, Y_train,
		#                     batch_size=batch_size, nb_epoch=nb_epoch,
		#                     verbose=1, validation_data=(X_test, Y_test))
		# score = model.predict_proba(X_test)
		# print score

def main():
	instance=Keras_example(Config())
	instance.classifier()
	pass

if __name__ == '__main__':
 	main() 

		

