
import numpy as np
import os

from six.moves import cPickle as pickle
import cv2

dirs = ['dataset/yawnMouth', 'dataset/normalMouth']
countYawn = 2481
countNormal = 16393


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#Generating dataset
def generate_dataset():
   
    maxX = 224
    maxY = 224

    i = 0
    j = 0
    pos = 0
    dataset = np.ndarray([countYawn + countNormal, maxY, maxX, 1], dtype='float32')
#reading dataset   
    for dir in dirs:
        for filename in os.listdir(dir):
            if (filename.endswith('.jpg') or filename.endswith('.JPG')):
                im = cv2.imread(dir + '/' + filename)
                im = cv2.resize(im, (maxX, maxY))
                im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
                dataset[i, :, :, :] = im[:, :, :,:]
                i += 1
        if pos == 0:
            labels = np.ones([i, 1], dtype=int)
            j = i
            pos += 1
        else:
            labels = np.concatenate((labels, np.zeros([i - j, 1], dtype=int)))
    print('Data set', dataset.shape, labels.shape)
    return dataset, labels


dataset, labels = generate_dataset()
print("Total = ", len(dataset))
dataset,labels=unison_shuffled_copies(dataset,labels)
dataset,labels=unison_shuffled_copies(dataset,labels)
totalCount = countYawn + countNormal
split = int(countYawn*0.6)
splitEnd = countYawn
split2 = countYawn + int(countNormal * 0.6)

#adding yawn samples
train_dataset = dataset[:split]
train_labels = np.ones([split, 1], dtype=int)
test_dataset = dataset[split:splitEnd]
test_labels = np.ones([splitEnd - split, 1], dtype=int)

#adding normal samples
train_dataset = np.concatenate((train_dataset, dataset[splitEnd:split2]))
train_labels = np.concatenate((train_labels, np.zeros([split2 - splitEnd, 1], dtype=int)))
test_dataset = np.concatenate((test_dataset, dataset[split2:]))
test_labels = np.concatenate((test_labels, np.zeros([totalCount - split2, 1], dtype=int)))

#shuffling dataset
train_dataset,train_labels=unison_shuffled_copies(train_dataset,train_labels)
test_dataset,test_labels=unison_shuffled_copies(test_dataset,test_labels)

#saving in pickle file

pickle_file = 'yawn_mouths.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
