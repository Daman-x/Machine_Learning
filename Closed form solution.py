#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston # builtin dataset


# In[3]:


bot = load_boston()
X = bot.data
y = bot.target


# In[4]:


X.shape


# In[5]:


a = np.linalg.inv(np.dot(X.T,X)) 
b = np.dot(a,X.T)
theta = np.dot(b,y)


# In[6]:


theta


# In[8]:


def hpy(X,theta):
    y_ = np.dot(X,theta)
    return y_


# In[9]:


haty = hpy(X,theta)


# In[12]:


def r2score(y,haty):
    mean = np.mean(y)
    num = np.sum((y-haty)**2)
    denum = np.sum((y-mean)**2)
    sum = 1-num/denum
    return sum*100


# In[13]:


r2score(y,haty)


# In[ ]:




