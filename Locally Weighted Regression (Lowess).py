#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


np.random.seed(67)
X = 2 - 3 * np.random.normal(0, 1, 20)
Y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 20)  # RANDOM DATA MADE
plt.scatter(X,Y, s=10)
plt.show()


# In[5]:


ones = np.ones((m,1))
X = X.reshape((20,1))
X = np.hstack((ones,X))


# In[3]:


# (XWXT)`1 XT W Y
m = X.shape[0]
X.shape


# In[4]:


u = np.mean(X)
std = np.std(X)
X = (X - u)/std


# In[6]:


def calweg(X,querypoint,tau = 5):
    m = X.shape[0]
    w = np.eye(m)
    x = querypoint
    for i in range(m):
        xi = X[i]
        w[i][i] = np.exp(np.dot( (xi-x),(xi-x).T)/ -2*tau**2)
    return w


# In[7]:


w = calweg(X,[1,4.5])


# In[1]:


#X = X.reshape((20,1))


# In[8]:


w.shape


# In[9]:


a = np.linalg.inv( np.dot( ( np.dot(X.T,w) ), X) )


# In[10]:


a.shape


# In[11]:


b = np.dot( (np.dot(a,X.T)), w )


# In[12]:


b.shape


# In[13]:


theta = np.dot(b,Y)


# In[3]:


#theta


# In[15]:


X.shape


# In[16]:


def lowess(X,y):
    m = X.shape[0]
    test = X
    y_ = []
    for i in range(m):
        w= calweg(test,test[i])
        a = np.linalg.inv( np.dot(( np.dot(X.T,w) ), X))
        b = np.dot( (np.dot(a,X.T)), w)
        theta = np.dot(b,Y)
        hat = theta[1] * X[i,1] + theta[0]
        y_.append(hat)
        
    return y_
        
    


# In[17]:


y_ = lowess(X,Y)


# In[18]:


theta.shape


# In[19]:


theta[0]


# In[20]:


theta[1]


# In[21]:


y_


# In[22]:


plt.scatter(X[:,1],Y)
plt.scatter(X[:,1],y_,color='red')
plt.plot()


# In[ ]:




