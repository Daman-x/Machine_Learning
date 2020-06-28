#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[6]:


X,y = make_blobs(n_samples=500,n_features=2,centers=5, random_state=67)


# In[183]:



plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[184]:


cluster = {}
k=5
color = ["orange","pink","green","yellow","red"]

for i in range(k):
    points = []
    center = 7*(2*(np.random.random((X.shape[1],)))-1)
    clusters = {
        
        "center" : center,
        "points" : points,
        "color" :color[i]
    }
    cluster[i] = clusters


# In[185]:


cluster


# In[186]:


def distance(a,b):
    dist = np.sqrt(np.sum((a-b)**2))
    return dist


# In[187]:


def assignpoints(X,cluster):
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            a = X[i]
            b = cluster[j]['center']
            dis = distance(a,b)
            dist.append(dis)
        index = np.argmin(dist)
        cluster[index]['points'].append(X[i])
    


# In[188]:


def updatepoints(cluster):
    for i in range(k):
        pts = np.array(cluster[i]['points'])
        if pts.shape[0] > 0 :
            cluster[i]['center'] = np.mean(pts,axis=0)
            cluster[i]['points'] = []
        


# In[189]:


def ploting(cluster):
      for i in range(k):
            a = np.array( cluster[i]['center'])
            b = np.array (cluster[i]['points'])
            
           # color = cluster[i]['color']
            try:
                plt.scatter(b[:,0],b[:,1])
            except:
                    pass
            plt.scatter(a[0],a[1],color='black',marker='*')


# In[190]:


assignpoints(X,cluster)
ploting(cluster)


# In[19]:


center = 10*((2*(np.random.random(X.shape[1],)))-1)


# In[20]:


center


# In[191]:


updatepoints(cluster)
assignpoints(X,cluster)
ploting(cluster)


# In[192]:


updatepoints(cluster)
assignpoints(X,cluster)
ploting(cluster)


# In[193]:


updatepoints(cluster)
assignpoints(X,cluster)
ploting(cluster)


# In[194]:


updatepoints(cluster)
assignpoints(X,cluster)
ploting(cluster)


# In[ ]:





# In[ ]:




