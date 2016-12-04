import numpy as np
from scipy import io, stats
from scipy.signal import resample, correlate
from sklearn import preprocessing
sample_size = 600
import os
from scipy.stats import pearsonr
import pandas as pd

def get_data(fname_prefix, num_segments, safe_dict):
	files = []
	preictal = 0
	interictal = 0
	input_dataset = np.array([])
	labels = np.array([])
	for i in range(1, num_segments):
		fname = fname_prefix + str(i) + '_' + str(0) + '.mat'
		if os.path.isfile(fname) and safe_dict[fname[8:]]:
			files.append(fname)
			data_struct = io.loadmat(fname)['dataStruct']
			data = data_struct[0][0][0].transpose()
			
			correlations = np.array([])
			for j in range(16):
				for k in range(j+1, 16):
					#corr = correlate(resampled_data[i], resampled_data[j], mode = 'same')
					corr = pearsonr(data[j], data[k])[0]
					if np.isnan(corr):
						corr = 0
					correlations = np.concatenate([correlations, [corr]])
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
			if len(input_dataset) > 0:
				#input_dataset = np.vstack((input_dataset, resampled_data))
				input_dataset = np.vstack((input_dataset, data_stats_list))
				#print input_dataset.shape
			else:
				#input_dataset = resampled_data
				input_dataset = data_stats_list
			labels = np.append(labels, 0)
			interictal+=1
		fname = fname_prefix + str(i) + '_' + str(1) + '.mat'
		if os.path.isfile(fname) and safe_dict[fname[8:]]:
			#print(fname)
			files.append(fname)
			data_struct = io.loadmat(fname)['dataStruct']
			correlations = np.array([])
			data = data_struct[0][0][0].transpose()
			# Get correlations between each pair of channels.
			for j in range(16):
				for k in range(j+1, 16):
					#corr = correlate(resampled_data[i], resampled_data[j], mode = 'same')
					corr = pearsonr(data[j], data[k])[0]
					if np.isnan(corr):
						corr = 0
					correlations = np.concatenate([correlations, [corr]])
			
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
			#fourier = abs(np.fft.rfft(data)).T[:100].T
			#fourier = fourier.reshape(16*100)
			data_stats_list = np.hstack((data_stats_list, correlations))
			if len(input_dataset) > 0:e
				input_dataset = np.vstack((input_dataset, data_stats_list))
			else:
				input_dataset = data_stats_list
			labels = np.append(labels, 1)
			preictal+=1
	print(len(files))
	print(interictal)
	print(preictal)
	print(input_dataset.shape)
	print(labels.shape)
	return input_dataset, labels, files

def get_test_data(fname_prefix, num_segments):
	files = []
	input_dataset = np.array([])
	for i in range(1, num_segments+1):
		fname = fname_prefix + str(i) + '.mat'
		#want to check for both _0 and _1 filename
		if os.path.isfile(fname):
			files.append(fname)
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
			if len(input_dataset) > 0:
				input_dataset = np.vstack((input_dataset, data_stats_list))
				#print input_dataset.shape
			else:
				#input_dataset = resampled_data
				input_dataset = data_stats_list
	print(len(files))
	print(input_dataset.shape)
	return input_dataset

p = pd.read_csv('train_and_test_data_labels_safe.csv')
p = (p.set_index('image')).drop('class', axis=1)
safe_dict = p.to_dict()['safe']
print safe_dict

input_dataset, labels, files= get_data('train_1/1_', 1152, safe_dict)
print(input_dataset.shape)
data_dict = {'data' : input_dataset, 'labels': labels, 'filenames': files}
io.savemat('train1.mat', data_dict)


input_dataset, labels, files = get_data('train_2/2_', 2196, safe_dict)
print(input_dataset.shape)
data_dict = {'data' : input_dataset, 'labels': labels, 'filenames': files}
io.savemat('train2.mat', data_dict)


input_dataset, labels, files = get_data('train_3/3_', 2244, safe_dict)
print(input_dataset.shape)
data_dict = {'data' : input_dataset, 'labels': labels, 'filenames': files}
io.savemat('train3.mat', data_dict)


input_dataset = get_test_data('test_1_new/new_1_', 216)
print(input_dataset.shape)
data_dict = {'data' : input_dataset}
io.savemat('test1.mat', data_dict)


input_dataset = get_test_data('test_2_new/new_2_', 1002)
print(input_dataset.shape)
data_dict = {'data' : input_dataset}
io.savemat('test2.mat', data_dict)


input_dataset = get_test_data('test_3_new/new_3_', 690)
print(input_dataset.shape)
data_dict = {'data' : input_dataset}
io.savemat('test3.mat', data_dict)