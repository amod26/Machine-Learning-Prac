#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


column_names = ['user_id','item_id','rating', 'timestamp'
]


# In[4]:


df = pd.read_csv('/Users/senorpete/Desktop/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Recommender-Systems/u data.csv', sep='\t', names=column_names)


# In[5]:


df.head()


# In[6]:


movie_titles = pd.read_csv('/Users/senorpete/Desktop/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Recommender-Systems/mtitles.csv')


# In[7]:


movie_titles.head()


# In[32]:


df= pd.merge(df,movie_titles, on='item_id')


# In[33]:


df.head()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


sns.set_style('white')


# In[52]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[53]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[54]:


ratings = df.groupby('title')['rating'].mean()


# In[55]:


ratings.head()


# In[76]:


ratings['no_ratings']=pd.DataFrame(df.groupby('title')['rating'].count())


# In[77]:


ratings.head()


# In[84]:


ratings['no_ratings'].hist(bins=70)


# In[111]:


df.groupby('title')['rating'].mean().hist(bins=70)


# In[120]:


moviemat = df.pivot_table(index = 'user_id', columns = 'title', values= 'rating' )


# In[128]:


moviemat.head()


# In[137]:


ratings.head(20).sort_values(ascending=False)


# In[140]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']


# In[142]:


starwars_user_ratings.head()


# In[144]:


liarliar_user_ratings.head()


# # Corrwith

# In[146]:


similar_to_starwars= moviemat.corrwith(starwars_user_ratings)


# In[147]:


similar_to_liarliar= moviemat.corrwith(liarliar_user_ratings)


# In[181]:


corr_starwars= pd.DataFrame(similar_to_starwars,columns= ['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# In[166]:


# movies with same correlation as Star_wars
corr_starwars.sort_values('Correlation', ascending=False).head(10)


# In[169]:


corr_starwars.head()


# In[154]:


# filtering movies with less than 100 movie reviews


# In[172]:


corr_starwars[corr_starwars['rating']>100].sort_values('Correlation',ascending=False).head()


# In[176]:


corr_liarliar= pd.DataFrame(similar_to_liarliar,columns= ['Correlation'])
corr_liarliar.dropna().head()


# In[182]:


corr_liarliar= corr_liarliar.join(ratings['no_of_ratings'])


# In[184]:


corr_liarliar.dropna(inplace=True)
corr_liarliar.head()


# In[185]:


corr_liarliar[corr_liarliar['rating']>100].sort_values('Correlation', ascending=False).head(10)


# In[ ]:




