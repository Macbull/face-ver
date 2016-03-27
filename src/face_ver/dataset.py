import os
from scipy import ndimage
from six.moves import cPickle as pickle
import numpy as np
import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

image_size = [112,92]
pixel_depth = 255.0
num_channels = 1
root = 'att_faces'
data_folders = [
	os.path.join(root, d) for d in sorted(os.listdir(root))
	if os.path.isdir(os.path.join(root, d))]

def load_letter(folder, min_num_images):
	"""Load the data for a single letter label."""
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape = (len(image_files), image_size[0], image_size[1]), dtype=np.float32)
	image_index = 0
	print(folder)
	for image in os.listdir(folder):
		if not image.startswith('.') and os.path.isfile(os.path.join(folder, image)):
			image_file = os.path.join(folder,image)
			try:
				image_data = (ndimage.imread(image_file).astype(float)*2 - pixel_depth) /pixel_depth # !!!!should be pixel_depth*2 --fixed
				if image_data.shape != (image_size[0], image_size[1]):
					raise Exception('Unexpected image shape: %s' %str(image_data.shape))
				dataset[image_index, :, :] = image_data
				image_index +=1
			except IOError as e:
				print('Could not read:', image_file, ':', e, '- it\'s ok, skipping')
			
	num_images = image_index
	dataset = dataset[0:num_images, :, :]
	if num_images < min_num_images:
		raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))
	print('FUll dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset

def maybe_pickle(data_folders, min_num_images_per_class = 10, force=False):
	dataset_names = []
	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
			# YOu mayy override by setting force=True.
			print('%s already present - Skipping pickling.' % set_filename)
		else:
			print('Pickling %s.' % set_filename)
			dataset = load_letter(folder, min_num_images_per_class)
			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)
				
	return dataset_names

train_folders = maybe_pickle(data_folders[0:30])
test_folders = maybe_pickle(data_folders[30:40])


def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size[0], img_size[1]), dtype = np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels

def merge_datasets(pickle_files, train_size, num_classes, valid_size=0):
	
	valid_dataset, valid_labels = make_arrays(valid_size, image_size)
	train_dataset, train_labels = make_arrays(train_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes

	start_v, start_t = 0,0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				np.random.shuffle(letter_set)
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class

				train_letter = letter_set[vsize_per_class:end_l, :, :]
				train_dataset[start_t:end_t,:,:] = train_letter
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise

	return valid_dataset, valid_labels, train_dataset, train_labels
	
train_size = 10*7
# train_size = 200000
valid_size = 10*3
test_size = 10*10

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_folders, train_size, 30, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_folders, test_size, 10)
print('\n Merging \n')
print ('Training: ', train_dataset.shape, train_labels.shape)
print ('Validation: ', valid_dataset.shape, valid_labels.shape)
print('Testing: ', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


def save_dataset(pickle_file, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	try:
		f = open(pickle_file, 'wb')
		save = {
		'train_dataset': train_dataset,
		'train_labels': train_labels,
		'valid_dataset': valid_dataset,
		'valid_labels': valid_labels,
		'test_dataset': test_dataset,
		'test_labels': test_labels,
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise
	return pickle_file

dataset_file = save_dataset('att_faces.pickle', train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

def open_file(pickle_file):
	with open(pickle_file,'rb') as f:
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		train_labels = save['train_labels']
		valid_dataset = save['valid_dataset']
		valid_labels = save['valid_labels']
		test_dataset = save['test_dataset']
		test_labels = save['test_labels']
		del save
		print('\n File opened\n')
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test dataset', test_dataset.shape, test_labels.shape)

open_file(dataset_file)

def reformat(dataset, labels):
	# data as a flat matrix
	dataset = dataset.reshape((-1, image_size[0], image_size[1], num_channels)).astype(np.float32)
	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('\nDataset reformatted\n')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test dataset', test_dataset.shape, test_labels.shape)

def pair(dataset, labels):
	pairs = np.ndarray(shape = (2, nCr(dataset.shape[0],2), dataset.shape[1], dataset.shape[2], dataset.shape[3]), dtype=np.float32)
	pair_labels = np.ndarray(shape = (nCr(labels.shape[0],2)), dtype=np.float32)

	count = 0
	for i in range(dataset.shape[0]):
		for j in range(i+1,dataset.shape[0]):
			pairs[0,count,:,:,:] = dataset[i,:,:,:]
			pair_labels[count] = 1*(labels[i]!=labels[j])
			pairs[1,count,:,:,:] = dataset[j,:,:,:]
			count +=1
	return pairs, pair_labels

train_dataset, train_labels = pair(train_dataset, train_labels)
valid_dataset, valid_labels = pair(valid_dataset, valid_labels)
test_dataset, test_labels = pair(test_dataset, test_labels)
print('\npaired\n')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test dataset', test_dataset.shape, test_labels.shape)
paired_dataset = save_dataset('att_faces_paired.pickle', train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
