
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.regularizers import l2
from keras.utils import plot_model


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pydot
import graphviz

from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


# In[3]:


path = './resize'
imlist = os.listdir(path)

num_samples = size(imlist)
print(num_samples)


# In[4]:


immatrix = array([array(Image.open('./resize/' + im)).flatten()
                 for im in imlist], 'f')


# In[5]:


print(immatrix)


# In[6]:


label = np.ones((num_samples, ), dtype = int)

label[0:99] = 0
label[100:199] = 1
label[200:299] = 2
label[300:399] = 3
label[400:499] = 4
label[500:599] = 5
label[600:699] = 6
label[700:799] = 7
label[800:899] = 8
label[900:999] = 9


# In[7]:


data, Label = shuffle(immatrix, label, random_state = 2)
train_data = [data, Label]


# In[8]:


img_rows = 32
img_cols = 32


# In[9]:


print(train_data[0].shape)
print(train_data[1].shape)


# In[10]:


(X, y) = (train_data[0], train_data[1])


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 4)


# In[12]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[13]:


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)


# In[14]:


X_train /= 255
X_test /= 255


# In[15]:


Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# In[16]:


print(X_train.shape)


# In[17]:


model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1), padding='same', kernel_regularizer=l2(0.001)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

#model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
#model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
#model.add(MaxPooling2D())
#model.add(Dropout(0.3))

#model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
#model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
#model.add(MaxPooling2D())
#model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[18]:


opt = 'adam'
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])


# In[19]:


hist = model.fit(X_train, Y_train, epochs = 30, batch_size = 20, validation_data = (X_test, Y_test), verbose = 2)


# In[20]:


plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()


# In[21]:


for i in range(10):
    prepath = './myvoice/' + str(i) + '_crop.png'
    predictmatrix = array([array(Image.open(prepath)).flatten()], 'f')
    predictmatrix = predictmatrix.reshape(1, 32, 32, 1)
    prearr = model.predict(x = predictmatrix, batch_size = 32, verbose = 0)
    print('prediction : ', prearr, ' data : ', i)

