#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


univ = pd.read_csv("Universities.csv")
univ


# In[4]:


univ.describe()


# In[7]:


Univ1 = univ.iloc[:,1:]


# In[8]:


Univ1


# In[9]:


cols = Univ1.columns


# In[13]:


from sklearn.preprocessing import standerdscaler
scaler = StanderdScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols )
scaled_Univ_df


# In[ ]:




