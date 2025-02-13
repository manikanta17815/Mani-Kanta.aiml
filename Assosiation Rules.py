#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# ### Observations
# - There are no null values
# - All objects are categorical in nature
# - As the columns are categorical,we can adopt one-hot-encoding

# In[5]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index,counts.values)


# In[7]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index,counts.values)


# In[8]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index,counts.values)


# In[9]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index,counts.values)


# In[11]:


# Perform one-hot encoding on categorical columns
df =pd.get_dummies(titanic,dtype=int)
df.head()


# In[12]:


df.info()


# In[15]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True, max_len=None)
frequent_itemsets


# In[16]:


frequent_itemsets.info()


# In[17]:


rules = association_rules(frequent_itemsets, metric="lift",min_threshold=1.0)
rules


# In[18]:


rules.sort_values(by='lift', ascending = False)


# In[20]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7)
plt.show()


# In[ ]:




