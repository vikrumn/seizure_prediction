import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn import preprocessing
from sklearn import linear_model, feature_selection, ensemble, model_selection
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import os
import csv

"""
Combines training data from all three patients, then trains logistic 
and random forest classifiers on the data, runs both on the test data, and 
takes the average class probabilities.
"""

train = io.loadmat('train1.mat')

print train['data'].shape
train['labels'] = np.ravel(train['labels'])
train['labels'] = train['labels'].transpose()
print train['labels'].shape

train2 = io.loadmat('train2.mat')
train3 = io.loadmat('train3.mat')
print train2['data'].shape
train['data'] = np.vstack((train['data'], train2['data']))
train['data'] = np.vstack((train['data'], train3['data']))
print train['labels'].shape
print train2['labels'].shape
train_test = io.loadmat('old_test.mat')
train['data'] = np.vstack((train['data'], train_test['data']))
train2['labels'] = np.ravel(train2['labels'])
train3['labels'] = np.ravel(train3['labels'])
train_test['labels'] = np.ravel(train_test['labels'])
train['labels'] = np.hstack((train['labels'], train2['labels']))
train['labels'] = np.hstack((train['labels'], train3['labels']))
train['labels'] = np.hstack((train['labels'], train_test['labels']))
"""
old_test = io.loadmat('old_test3.mat')
old_test['labels'] = np.ravel(old_test['labels'])
train['data'] = np.vstack((train['data'],old_test['data']))
print train['labels'].shape
print old_test['labels'].shape
train['labels'] = np.hstack((train['labels'],old_test['labels']))
"""
random_state = np.random.get_state()
np.random.shuffle(train['data'])
np.random.set_state(random_state)
np.random.shuffle(train['labels'])

kf = model_selection.KFold(n_splits = 10)
logistic = linear_model.LogisticRegression(C=1e5)
rf = ensemble.RandomForestClassifier()

#Train both classifiers using 10-fold cross validation
for train_index, test_index in kf.split(train['data'], train['labels']):
	train_data = train['data'][train_index]
	train_labels = train['labels'][train_index]
	validation_data = train['data'][test_index]
	validation_labels = train['labels'][test_index]
	logistic.fit(train_data, train_labels)
	rf.fit(train_data, train_labels)
	print rf.score(validation_data, validation_labels)
	print logistic.score(validation_data, validation_labels)

test = io.loadmat('test1.mat')
preds = logistic.predict_proba(test['data'])[:,1]
rf_preds = rf.predict_proba(test['data'])[:,1]
preds = (preds + rf_preds)/2
print preds
print np.sum(preds)
with open('test_labels1.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(preds)

test = io.loadmat('test2.mat')
preds = logistic.predict_proba(test['data'])[:,1]
rf_preds = rf.predict_proba(test['data'])[:,1]
preds = (preds + rf_preds)/2
print preds
print np.sum(preds)
with open('test_labels2.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(preds)

test = io.loadmat('test3.mat')
preds = logistic.predict_proba(test['data'])[:,1]
rf_preds = rf.predict_proba(test['data'])[:,1]
preds = (preds + rf_preds)/2
print preds
print np.sum(preds)
with open('test_labels3.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(preds)