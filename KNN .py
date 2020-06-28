#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# In[2]:


data = load_digits()
X = data.images
y = data.target
X[0].shape


# In[3]:


X.shape
lis = []


# In[5]:


plt.imshow(X[6],cmap=plt.get_cmap('gray')) 


# In[6]:


X.shape


# In[7]:


def distance (x,xi):
    dist = np.sqrt(np.sum((x - xi)**2))
    return dist


# In[8]:


def KNN(X,y,point,k=5):
    val = []
    m = X.shape[0]
    x = point.reshape((64,))
    for i in range(m):
        xi = X[i]
        xi = xi.reshape((64,))
        dist = distance(x , xi)
        val.append((dist,y[i]))
        
    vall = sorted(val,key= lambda x:x[0])[:k]
    val = np.array(vall)
    val_point = np.unique(val[:,1], return_counts = True)
    index = val_point[1].argmax()
    result = val_point[0][index]
    return result


# In[13]:


ypred = KNN(X,y,X[6]) # giving 6 no of pic to detect


# In[14]:


ypred  # pred which pic is this


# In[ ]:




