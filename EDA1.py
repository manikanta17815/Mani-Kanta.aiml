#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


data.dtypes


# In[7]:


data.describe()


# In[8]:


print(type(data))
print(data.shape)


# In[11]:


data1 = data.drop(['Unnamed: 0',"Temp C"],axis = 1)
data1


# In[12]:


data.info


# In[14]:


data1["Month"]=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[15]:


data1[data1.duplicated()]


# In[ ]:




