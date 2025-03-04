#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df


# In[8]:


X = df.drop(columns=['target'])
y = df['target']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[14]:


predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)


# In[ ]:





# In[18]:


# Prediction
data_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(data_input)
predicted_class = data.target_names[prediction[0]]

st.subheader("Prediction")
st.write(f"Predicted Class: **{predicted_class}**")


# In[ ]:




