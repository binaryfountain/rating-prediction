#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
dt = pd.read_csv('/home/murali/Desktop/rating_data/consol_v1.csv')


# In[32]:


dt['CONTENT'].value_counts()


# In[33]:


dt.columns=['DOCUMENT_ID','CONTENT_y','CONTENT_x']


# In[34]:


dt_rt_5 = dt[dt['CONTENT_y'].isin(['Overall Rating 5'])]
dt_rt_4 = dt[dt['CONTENT_y'].isin(['Overall Rating 4'])]


# In[35]:


dt_rt_5['count'] = dt_rt_5['CONTENT_x'].str.split().str.len()
print(dt_rt_5.shape)
dt_rt_5 = dt_rt_5[dt_rt_5['count'] > 50]
print(dt_rt_5.shape)
dt_rt_5.drop(['count'],axis=1,inplace=True)


# In[36]:


dt_rt_4['count'] = dt_rt_4['CONTENT_x'].str.split().str.len()
print(dt_rt_4.shape)
dt_rt_4 = dt_rt_4[dt_rt_4['count'] > 40]
print(dt_rt_4.shape)
dt_rt_4.drop(['count'],axis=1,inplace=True)


# In[37]:


dt_rt_4 = dt_rt_4.sample(n=4500)
dt_rt_5 = dt_rt_5.sample(n=10500)


# In[38]:


dt_rt_4 = dt_rt_4.append(dt_rt_5)


# In[39]:


dt = dt_rt_4.sample(frac=1)


# In[40]:


dt.shape


# In[41]:


dt.to_csv('/home/murali/Desktop/rating_4_5.csv',index=False)

