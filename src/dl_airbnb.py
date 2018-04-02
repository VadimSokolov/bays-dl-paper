#!/usr/bin/env python
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import regularizers
import xgboost as xgb

# ndcg5
def ndcg5(preds, y_true):
	k = 5
	n = len(y_true)
	num_class = preds.shape[1]
	index = np.argsort(preds, axis=1)
	top = index[:, -k:][:,::-1]
	rel = (np.reshape(y_true, (n, 1))==top).astype(int)
	cal_dcg = lambda y: sum((2**y - 1)/np.log2(range(2, k+2)))
	ndcg = np.mean((np.apply_along_axis(cal_dcg, 1, rel)))
	return 'ndcg5', -ndcg


print("Loading Data...")
# with open("../data/airbnb/X.pickle", 'rb') as fh:
with open("/Users/vsokolov/Downloads/X5k.pickle", 'rb') as fh:
	X = pickle.load(fh)
# with open("../data/airbnb/y.pickle", 'rb') as fh:
with open("/Users/vsokolov/Downloads/y5k.pickle", 'rb') as fh:
	y = pickle.load(fh)

# n = X.shape[0]
# ind_train = np.random.choice(range(n), size=int(0.8*n))

SS = preprocessing.StandardScaler()
X_std = SS.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_std, y, test_size=0.1, random_state=0)
label_train = np_utils.to_categorical(y_train)
label_val = np_utils.to_categorical(y_val)
num_class = max(y)+1



# a linear stack of layers
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1]),,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(Activation('relu'))
#model.add(Dropout(0.05))

model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.05))

model.add(Dense(num_class))
model.add(Activation('softmax'))

# multiclass logloss
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

# fit
# a typical minibatch size is 256
# shuffle the samples at each epoch
model.fit(X_train, label_train, batch_size=128, nb_epoch=20, validation_data=(X_val, label_val), shuffle=True, verbose=2)

preds = model.predict_proba(X_val, verbose=0)
ndcg5(preds, y_val)
