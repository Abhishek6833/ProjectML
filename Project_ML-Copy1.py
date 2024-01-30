#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[3]:


data=pd.read_csv("complaints.csv",nrows=100)


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


print(data)


# In[8]:


data.describe()


# In[9]:


data.isnull()


# In[10]:


data.isnull().sum()


# In[11]:


data.dtypes


# In[12]:


prod_data=data.groupby(["Complaint ID"])[['Date received','Product']].sum()


# In[13]:


prod_data


# In[14]:


x=data.iloc[:,:1].values
y=data.iloc[:,1:2].values
x_train,x_test,y_train,y_test=train_test_split(
    x,y, 
    train_size = 0.80, 
    random_state = 1)


# In[15]:


model=LogisticRegression(solver='lbfgs',max_iter=1000)


#     sd

# In[19]:


data['Date received_column'] = pd.to_datetime(data['Date received_column'])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[ ]:




