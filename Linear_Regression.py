#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


data = pd.read_csv('C:\\Users\\Daman\\Machine Learning\\data_0.csv')


# In[19]:


X = np.array(data['x'])
y = np.array(data['y'])




# In[20]:


n = X.mean()
std = X.std()
X = (X-n)/std


# In[ ]:





# In[22]:


plt.scatter(X,y)


# In[33]:


m = X.shape[0]  # Y = M X + C
 #m = theta 1
    #c = theta 0
 


# In[23]:


def error(X,y,theta):
    y_ = hpy(X,theta)
    error = np.sum((y_ - y)**2)  
    return error/m


# In[24]:


def hpy(X,theta):
    y_ = theta[1]*X + theta[0]
    return y_


# In[36]:


def gradient(X,y,theta):
    grad = np.zeros((2,))
    y_ = hpy(X,theta)
    grad[0] = np.sum((y_-y))
    grad[1] = np.dot( X.T,(y_ - y))
    return grad/m 


# In[37]:


def gradientdescent(X,y, epcho = 200,learning_rate = 0.1):
    grad = np.zeros((2,))
    theta = np.zeros((2,))
    err = []
    theta_l = []
    
    for i in range(epcho):
        e = error(X,y,theta)
        err.append(e)
        grad = gradient(X,y,theta)
        theta = theta - learning_rate * grad 
        theta_l.append(theta)
        
    return err,theta,theta_l 
        


# In[38]:


err,theta,theta_l = gradientdescent(X,y)


# In[41]:


plt.scatter(X,y)    
y_ = theta[1] * X + theta[0]
plt.plot(X,y_,"red")


# In[42]:


def r2score():
    Ypred = hpy(X,theta)
    num = np.sum((Ypred-y)**2)
    demo = np.sum((y-y.mean())**2)
    sum = (1-num/demo)
    return sum*100


# In[43]:


r2score() // shows the accuracy of the line


# In[44]:


#plt.ion()
for i in range(0,100,2):
    y_ = theta_l[i][1]*X + theta_l[i][0]
    y__ = theta[1] * X + theta[0]
    plt.scatter(X,y)
    plt.plot(X,y_,"red")
    plt.plot(X,y__,"orange")
    plt.draw()
    plt.pause(1)
    #destroy the last object
    plt.clf()


# In[29]:


grad = np.zeros((2,))


# In[30]:


grad


# In[ ]:





# In[40]:


theta_l


# In[ ]:




