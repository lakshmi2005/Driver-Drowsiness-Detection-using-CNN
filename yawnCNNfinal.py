from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam
from keras import backend as K
from keras import regularizers
#from keras.applications.vgg16 import VGG16
from sklearn import svm

from six.moves import cPickle as pickle

#loading pickle file
pickle_files = ['yawn_mouths.pickle']
i = 0
for pickle_file in pickle_files:
    with open(pickle_file,'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
        else:
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
        del save 
    i += 1

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 32
nb_classes = 1
epochs = 20

#splitting into x-train and y_train 
X_train = train_dataset
X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])

Y_train = train_labels

X_test = test_dataset
X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])

Y_test = test_labels

# input image dimensions
_,img_channels, img_rows, img_cols = X_train.shape

K.set_image_dim_ordering("th")

#cnn architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(img_channels, img_rows, img_cols)))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.2))
model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001),activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes,kernel_initializer='normal'))
model.add(Activation('sigmoid'))

model.summary()

#model compilation
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history= model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=[X_test, Y_test],shuffle=True)

# model.save('yawnModel.h5')

score = model.evaluate(X_test, Y_test, verbose=1)
#plotting the model
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='center right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model training loss vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='center right')
plt.show()

print('Test score:', score[0])
print('Test accuracy:', score[1])

