#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[23]:


data = pd.read_csv('C:\\Users\\Daman\\Machine Learning\\mutlivar_data.csv')


# In[24]:


y = np.array(data['price'])
y


# In[26]:


X = np.array(data.values[ : ,  3  :  12  ],dtype = float)
X.shape


# In[27]:


m = X.shape[0]
n = X.shape[1]


# In[28]:


u = np.mean(X)
std = np.std(X)
X = (u - X)/std


# In[29]:


ones = np.ones((m,1))


# In[30]:


ones


# In[31]:


#X = X.reshape((20,1))
X = np.hstack((ones,X))
X.shape


# In[54]:


def gradientdescent(X,y,epoch = 500 , learning_rate = 0.2):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros((n,))
    grad = np.zeros((n,))
    err = []
    for i in range(epoch):
        e = error(X,y,theta)
        err.append(e)
        grad = gradient(X,y,theta)
        theta = theta - learning_rate * grad
    return err,theta


# In[33]:


def error(X,y,theta):
    m = X.shape[0]
    y_ = hpy(X,theta)
    error = np.sum((y_ - y)**2)
    return error/m


# In[34]:


def hpy(X,theta):
    y_ = np.dot(X,theta)
    return y_


# In[35]:


def gradient(X,y,theta):
    m = X.shape[0]
    n = X.shape[1]
    grad = np.zeros((n,))
    y_ = hpy(X,theta)
    grad = np.dot((y_- y).T,X)
    return grad/m


# In[55]:


err,theta = gradientdescent(X,y)


# In[56]:


ypred = hpy(X,theta)
ypred.shape


# In[57]:


def r2_score(y,ypred):
    ymean = y.mean()
    num = np.sum((y-ypred)**2)
    denum = np.sum((y-ymean)**2)
    score = 1 - num/denum
    return score*100


# In[ ]:





# In[58]:


r2_score(y,ypred)


# In[ ]:





# In[ ]:





# In[ ]:




