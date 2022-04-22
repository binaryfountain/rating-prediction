#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dat = pd.read_csv('/home/murali/Desktop/fifty_new_rating_4_5.csv')
dat.head(3)
dat.columns=['DOCUMENT_ID','RATING','CONTENT']


# In[2]:


dat['CONTENT'] = dat['CONTENT'].str.rstrip()
dat['CONTENT'] = dat['CONTENT'].str.lstrip()


# In[3]:


dat.head(3)


# In[4]:


dat['CON'] = dat['CONTENT'].apply(lambda x: ' '.join(x.split()[:500]))


# In[5]:


aa = dat['CON'].str.split('.')
aa.head(2)


# In[6]:


# aa[0]


# In[7]:


# aa[1]


# In[6]:


bb = pd.DataFrame(aa)


# In[7]:


cc = pd.DataFrame(bb.CON.tolist(), index= bb.index)


# In[10]:


# dd = cc.iloc [0:5, 0:5] 


# In[9]:


dd = cc


# In[ ]:





# In[10]:


from transformers import pipeline
classifier = pipeline('sentiment-analysis')


# In[13]:


# ll12 = dd[0].tolist()
# len(ll12)


# In[11]:


bb=pd.DataFrame()


# In[12]:


for i in range(0,dd.shape[1]):
    print('i value is ',i)
    ll12 = dd[i].tolist()
    results3=[]
    for j in range(len(ll12)):
        print('j value is ',j)
        if((ll12[j]==None) or (ll12[j]=='')):
            results2 = 'Blank'
        else: 
            results2 =  classifier(ll12[j]) 
        results3.append(results2)
        
        if results3[j] == 'Blank':
            results3[j] =  [{'label': 'Blank', 'score': 0.0}]
        
    aa = pd.DataFrame(results3)
    aa.columns = ['text']
    aa['text'] = aa['text'].astype(str)
    aa['label']= aa['text'].str.split(',').str[0].str.split(':').str[1]
    aa['score']= aa['text'].str.split(',').str[1].str.split(':').str[1].str.replace('}', '')
    aa['label'] = aa['label'].astype(str)
    aa['score'] = aa['score'].astype(float)
    bb[i] = aa['label']


# In[13]:


bb.shape


# In[14]:


bb.head(4)


# In[15]:


bb['combined']= bb.values.tolist()


# In[16]:


bb.head(4)


# In[17]:


zz = bb.combined.tolist()


# In[21]:


# t = []
# for i in range(0,len(zz)):
#     x = zz[i].count(" 'POSITIVE'")
#     y = zz[i].count(" 'NEGATIVE'")
#     z = zz[i].count(" 'Blank'")
#     m = x+y
#     s = [x,y,m,x/m,y/m,z]
#     t.append(s)


# In[18]:


t = []
for i in range(0,len(zz)):
    x = zz[i].count(" 'POSITIVE'")
    y = zz[i].count(" 'NEGATIVE'")
    z = zz[i].count(" 'Blank'")
    m = x+y
    s = y/m
    t.append(s)


# In[19]:


t


# In[20]:


dat['neg'] = t


# In[21]:


dat.head(5)


# In[22]:


encoded_dict = {'Overall Rating 1':1,'Overall Rating 2':2,'Overall Rating 3':3,'Overall Rating 4':4,'Overall Rating 5':5}
dat['label'] = dat.RATING.map(encoded_dict)


# In[23]:


dat.head(3)


# In[39]:


dat.to_csv('/home/murali/Desktop/count_neg_comment.csv',index=False)


# In[24]:


import numpy as np
X = dat['neg']
y = dat['label']
XX = np.array(X)
yy = np.array(y)


# In[ ]:





# In[25]:


# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2, random_state=1)

# X_train = X_train.reshape(1, -1)
# X_test = X_test.reshape(1, -1)

# printing the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)
 
# printing the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)


# In[ ]:





# In[26]:


X_train1 = X_train.reshape(-1, 1)
X_test1 = X_test.reshape(-1, 1)
# y_train1 = y_train.reshape(-1, 1)
# y_test1 = y_testX.reshape(-1, 1)


# In[ ]:





# In[27]:


# training the model on training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train1, y_train)
 
# making predictions on the testing set
y_pred = knn.predict(X_test1)


# In[28]:


y_test1 = y_test.reshape(-1, 1)


# In[29]:


y_pred1 = y_pred.reshape(-1, 1)
y_pred


# In[30]:


me = pd.DataFrame() 
me['pred'] = y_pred
me['act'] = y_test


# In[31]:


data_crosstab = pd.crosstab(me['act'], 
                            me['pred'],
                                margins = False)
print(data_crosstab)


# In[32]:


(748+640) / me.shape[0]


# In[33]:


from sklearn.metrics import precision_recall_fscore_support as score

predicted = me['pred'].tolist()
y_test = me['act'].tolist()

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[34]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



print(precision_score(y_test, predicted, average="micro"))
print(recall_score(y_test, predicted, average="micro")) 
print(f1_score(y_test, predicted, average="micro"))
print('/n')
print(precision_score(y_test, predicted, average="macro"))
print(recall_score(y_test, predicted, average="macro"))  
print(f1_score(y_test, predicted, average="macro"))
print('/n')
print(precision_score(y_test, predicted, average="weighted"))
print(recall_score(y_test, predicted, average="weighted")) 
print(f1_score(y_test, predicted, average="weighted"))


# In[282]:


from sklearn.ensemble import RandomForestClassifier
RandomForestClfModel = RandomForestClassifier()
RandomForestClfModel.fit(X_train1,y_train)


# In[283]:


y_pred = RandomForestClfModel.predict(X_test1)


# In[285]:


me = pd.DataFrame() 
me['pred'] = y_pred
me['act'] = y_test


# In[275]:


(1831+178) / (1832+570+421+178)


# In[251]:


2401 / (2401 + 599)


# In[ ]:





# In[209]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(dat['neg'], dat['label'], test_size=0.25, random_state=142)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
prediction_results = knn.predict(X_test)
print(prediction_results)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:


# ll12 = dd[0].tolist()
# results3=[]
# for i in range(len(ll12)):
#     if((ll12[i]==None) or (ll12[i]=='')):
#         results2 = 'Blank'
#     else: 
#         results2 =  classifier(ll12[i]) 
#     results3.append(results2)


# In[87]:


# for i in range(len(results3)):
#     if results3[i] == 'Blank':
#         results3[i] =  [{'label': 'Blank', 'score': 0.0}]


# In[ ]:





# In[88]:


# aa = pd.DataFrame(results3)
# aa.columns = ['text']
# aa['text'] = aa['text'].astype(str)
# aa['label']= aa['text'].str.split(',').str[0].str.split(':').str[1]
# aa['score']= aa['text'].str.split(',').str[1].str.split(':').str[1].str.replace('}', '')
# aa['label'] = aa['label'].astype(str)
# aa['score'] = aa['score'].astype(float)
# aa['label'].value_counts()


# In[90]:


# bb=pd.DataFrame()
# bb[i] = aa['label']


# In[ ]:





# In[18]:


# from textblob import TextBlob
# cc[['polarity', 'subjectivity']] = cc[0].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))


# In[19]:


# cc.shape


# In[21]:


# cc.head(30)

