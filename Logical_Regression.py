#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
# bulitin data used


# In[23]:


#data = pd.read_csv('C:\\Users\\Daman\\Machine Learning\\Logical_regression.csv')
X,Y = make_blobs(n_samples=500,n_features=2,centers=2,random_state=1)


# In[31]:


#y = np.array(data['Sex'])
X


# In[40]:


#X = np.array(data.values[:,1:3])
#X[:,1]
one = np.ones((X.shape[0],1))
X = np.hstack((one,X))
X.shape


# In[42]:


plt.scatter(X[:,1],X[:,2],c=Y)


# In[43]:


def gradientdescent(X,Y,epcho =200 ,learning_rate = 0.1 ):
    n = X.shape[1]
    grad = np.zeros((n,))
    theta = np.zeros((n,))
    err =[]
    for i in range(epcho):
        e = error(X,Y,theta)
        err.append(e)
        grad = gradient(X,Y,theta)
        theta = theta - learning_rate * grad
    return err,theta
        
    


# In[57]:


def hpy(X,theta):
    z = np.dot(X,theta)
    y_ = 1.0/(1+np.exp(-z))
    return y_


# In[54]:


def error(X,Y,theta):
    y_ = hpy(X,theta)
    err = np.dot(-Y , np.log(y_)) - np.dot(1-Y , np.log(1-y_))
    return err/X.shape[0]


# In[51]:


def gradient(X,Y,theta):
    grad = np.zeros((X.shape[1],))
    y_ = hpy(X,theta)
    grad = np.dot(X.T,(Y - y_))
    return grad/X.shape[0]
    


# In[58]:


err , theta = gradientdescent(X,Y)


# In[59]:


err


# In[ ]:


x2 = theta[0] - theta[1]*X[:,1]/theta[2]


# In[66]:


plt.scatter(X[:,1],X[:,2],c =Y)
plt.plot(X[:,1],x2,color = 'red')


# In[67]:


X[:,1]


# In[68]:


x2


# In[ ]:




