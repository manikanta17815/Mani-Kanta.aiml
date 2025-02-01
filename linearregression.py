#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[11]:


data1 = pd.read_csv("NewspaperData.csv")
print(data1)


# In[12]:


data1.info()


# In[13]:


data1.describe()


# In[14]:


plt.figure(figsize=(6,3))
plt.title("plot for Daily Sales")
plt.boxplot(data1["daily"],vert = False)
plt.show()


# In[16]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# # Observations
# - There are no missing values
# - The daily column values appear to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the boxplots

# In[17]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[18]:


data1["daily"].corr(data1["sunday"])


# In[19]:


data1[["daily","sunday"]].corr()


# ### Observations
# - The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# - The corelation is strong positive with person's correlation coefficient od 0.958154

# In[20]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[23]:


model1.summary()


# In[ ]:




