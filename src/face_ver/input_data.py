from six.moves import cPickle as pickle
train_samples = 2415
class Dataset(object):
	def __init__(self, train_dataset, test_dataset, valid_dataset, train_labels, test_labels, valid_labels):
		self._train_dataset = train_dataset
		self._train_labels = train_labels
		self._test_dataset = test_dataset
		self._test_labels = test_labels
		self._valid_dataset = valid_dataset
		self._valid_labels = valid_labels

		self._image_size = (train_dataset.shape[2], train_dataset.shape[3], train_dataset.shape[4])
		self._train_samples = train_dataset.shape[1]
		self._test_samples = test_dataset.shape[1]
		self._valid_samples = valid_dataset.shape[1]
		train_samples = self._train_samples
	@property
	def train_dataset(self):
	    return self._train_dataset[0], self._train_dataset[1], self._train_labels
	
	@property
	def test_dataset(self):
	    return self._test_dataset[0], self._test_dataset[1], self._test_labels

	@property
	def valid_dataset(self):
		return self._valid_dataset[0], self._valid_dataset[1], self._valid_labels

	@property
	def image_size(self):
		return self._image_size

	@property
	def train_samples(self):
		return self._train_samples

	@property
	def test_samples(self):
		return self._test_samples

	@property
	def valid_samples(self):
		return self._valid_samples
    
		    
def read():	
	pickle_file = 'att_faces/att_faces_paired.pickle'
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
		datasets = Dataset(train_dataset, test_dataset, valid_dataset, train_labels, test_labels, valid_labels)
		return datasets
