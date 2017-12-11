
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from PIL import Image
from numpy import *

from sklearn.utils import shuffle


# In[10]:


path1 = './crop'
path2 = './resize'


# In[11]:


listing = os.listdir(path1)
num_samples = size(listing)


# In[14]:


for file in listing :
    im = Image.open(path1 + '/' + file)
    img = im.resize((32, 32))
    gray = img.convert('L')
    gray.save(path2 + '/' + file, "JPEG")

