#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('/Users/senorpete/Desktop/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/K-Nearest-Neighbors/Classified data.csv', index_col=0)


# In[4]:


df.head()


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


scaler= StandardScaler()


# In[7]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[8]:


scaled_features = scaler.transform((df.drop('TARGET CLASS',axis=1)))


# In[9]:


scaled_features


# In[10]:


df.feat= pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[11]:


df.feat.head()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X=df.feat
y=df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


knn= KNeighborsClassifier(n_neighbors=1)


# In[16]:


knn.fit(X_train,y_train)


# In[17]:


pred = knn.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix,classification_report


# In[19]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[23]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[24]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[25]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[26]:


# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# We were able to squeeze some more performance out of our model by tuning to a better K value!

# In[ ]:




