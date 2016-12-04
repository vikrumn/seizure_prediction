import numpy as np
from scipy import io, stats
from scipy.signal import resample, correlate
from sklearn import preprocessing
sample_size = 600
import os
from scipy.stats import pearsonr
import pandas as pd


def get_test_data(fname):
	if os.path.isfile(fname):
		correlations = np.array([])
		data_struct = io.loadmat(fname)['dataStruct']
		# Get the actual EEG data, consisting of 240000 x 16 matrix
		data = data_struct[0][0][0].transpose()
		
		for j in range(16):
			for k in range(j+1, 16):
				#corr = correlate(resampled_data[i], resampled_data[j], mode = 'same')
				corr = pearsonr(data[j], data[k])[0]
				if np.isnan(corr):
					corr = 0
				correlations = np.concatenate([correlations, [corr]])
		
		#resample new shape = (16,600)
		
		means = np.array([])
		stdevs = np.array([])
		data_stats_list = np.array([])
		for j in range(16):
			data_stats = np.array(stats.describe(data[j])[2:])
			data_stats_list = np.append(data_stats_list, data_stats)
			#means = np.append(means, np.mean(abs(data[j])))
			#means = np.append(means, np.mean(abs(np.fft.rfft(data[j]))))
			#stdevs = np.append(stdevs, np.std(data[j]))
			#stdevs = np.append(stdevs, np.std(abs(np.fft.rfft(data[j]))))
		
		data_stats_list = np.hstack((data_stats_list, correlations))
		return data_stats_list
	else:
		return None


p = pd.read_csv('old_test_data_filenames.csv')
old_test_data_filenames = ['1_8.mat'] + list(p['1_8.mat'])
print len(old_test_data_filenames)
print old_test_data_filenames
input_dataset1 = np.array([])
input_dataset2 = np.array([])
input_dataset3 = np.array([])
for filename in old_test_data_filenames:
	path = 'test_' + filename[0] + '/' + filename
	print path
	processed_data = get_test_data(path)
	if processed_data is None:
		print "ERROR"
	else:
		if int(filename[0]) == 1:
			if len(input_dataset1) > 0:
					#print input_dataset.shape
					#print resampled_data.shape
					#input_dataset = np.vstack((input_dataset, resampled_data))
					input_dataset1 = np.vstack((input_dataset1, processed_data))
					#print input_dataset.shape
			else:
				#input_dataset = resampled_data
				input_dataset1 = processed_data
		elif int(filename[0]) == 2:
			if len(input_dataset2) > 0:
					#print input_dataset.shape
					#print resampled_data.shape
					#input_dataset = np.vstack((input_dataset, resampled_data))
					input_dataset2 = np.vstack((input_dataset2, processed_data))
					#print input_dataset.shape
			else:
				#input_dataset = resampled_data
				input_dataset2 = processed_data
		elif int(filename[0]) == 3:
			if len(input_dataset3) > 0:
					#print input_dataset.shape
					#print resampled_data.shape
					#input_dataset = np.vstack((input_dataset, resampled_data))
					input_dataset3 = np.vstack((input_dataset3, processed_data))
					#print input_dataset.shape
			else:
				#input_dataset = resampled_data
				input_dataset3 = processed_data

labels = np.ones(len(input_dataset1))
data_dict = {'data' : input_dataset1, 'labels': labels}
print input_dataset1.shape
io.savemat('old_test1.mat', data_dict)

labels = np.ones(len(input_dataset2))
data_dict = {'data' : input_dataset2, 'labels': labels}
print input_dataset2.shape
io.savemat('old_test2.mat', data_dict)

labels = np.ones(len(input_dataset3))
data_dict = {'data' : input_dataset3, 'labels': labels}
print input_dataset3.shape
io.savemat('old_test3.mat', data_dict)