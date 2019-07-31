#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('/Users/senorpete/Desktop/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Linear-Regression/USA_Housing.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


sns.pairplot(df)


# In[9]:


sns.distplot(df['Price'])


# In[10]:


sns.heatmap(df.corr(),annot=True)


# In[11]:


df.columns


# In[12]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price']]


# In[13]:


y=df['Price']


# In[14]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


lm=LinearRegression()


# In[28]:


lm.fit(X_train,y_train)


# In[29]:


print(lm.intercept_)


# In[30]:


print(lm.coef_)


# In[31]:


cdf= pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[32]:


cdf


# 
# # Predictions

# In[45]:


predictions = lm.predict(X_test)


# In[46]:


predictions


# In[48]:


plt.scatter(y_test,predictions)


# In[50]:


sns.distplot((y_test-predictions))


# In[51]:


from sklearn import metrics


# In[52]:


metrics.mean_squared_error(y_test,predictions)


# In[53]:


metrics.mean_absolute_error(y_test,predictions)


# In[54]:


metrics.mean_squared_log_error(y_test,predictions)


# In[ ]:




