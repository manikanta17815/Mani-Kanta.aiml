#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans 


# #### Clustering - Divide the universities in to groups(Clusters)

# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


Univ.describe


# In[5]:


Univ1 = Univ.iloc[:,1:]


# In[6]:


Univ1


# In[7]:


cols = Univ1.columns


# In[8]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[9]:


from sklearn.cluster import KMeans 
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[10]:


clusters_new.labels_


# In[11]:


set(clusters_new.labels_)


# In[12]:


Univ['clusterid_new'] = clusters_new.labels_
Univ


# In[13]:


Univ.sort_values(by = "clusterid_new")


# In[14]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations
# - Cluster 2 appears to be the top rated universities cluster as the cut off score, Top10, SFRatio parameter mean values are highest
# - Cluster 1 appears to occupy the middle level rated univresities
# - Cluster 0 comes as the lower level rated universities

# In[15]:


Univ[Univ['clusterid_new' ]==0]


# In[18]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




