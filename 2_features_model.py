#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
neg = pd.read_csv('/home/murali/Desktop/fifty_new_rating_4_5.csv')
vec = pd.read_csv('/home/murali/Desktop/word_vec_new_rating_1_2.csv')


# In[2]:


neg.columns = ['DOCUMENT_ID','RATING','CONTENT']


# In[3]:


vec.head(2)
encoded_dict = {'Overall Rating 1':1,'Overall Rating 2':2,'Overall Rating 3':3,'Overall Rating 4':4,'Overall Rating 5':5}
neg['label'] = neg.RATING.map(encoded_dict)


# In[14]:


d1 = pd.merge(neg,vec,on='DOCUMENT_ID',how='inner')


# In[48]:


d1.drop(['RATING_x','CONTENT_x','CON','RATING_y','CONTENT_y'],axis=1,inplace=True)


# In[49]:


d1.head(2)


# In[15]:


import numpy as np
print(neg['label'].value_counts())
neg['label'] = np.where(neg['label']==1,0,1)
print(neg['label'].value_counts())


# In[51]:


import numpy as np
X = d1[['neg','word_count','cluster']]
y = d1['label']
XX = np.array(X)
yy = np.array(y)


# In[52]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2,random_state=109) # 70% training and 30% test


# In[53]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[54]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[59]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[14]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[58]:


model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[60]:


y_test


# In[61]:


me = pd.DataFrame() 
me['pred'] = y_pred
me['act'] = y_test


# In[62]:


data_crosstab = pd.crosstab(me['act'], 
                            me['pred'],
                                margins = False)
print(data_crosstab)


# In[ ]:





# In[ ]:





# In[7]:


vec1 = neg
vec1['CONTENT'] = vec1['CONTENT'].str.lower()


# In[8]:


l1 = ['children','child','baby','kid','boy','girl','born']  
regstr1 = '|'.join(l1)
l2 = ['covid','corona','omicron','virus']  
regstr2 = '|'.join(l2)
l3 = ['wait','waiting','waited']  
regstr3 = '|'.join(l3)
l4 = ['insurance','bill','charge','cover','claim','billing']  
regstr4 = '|'.join(l4)

l5 = ['answer','appointment']  
regstr5 = '|'.join(l5)
l6 = ['horriable','insane']  
regstr6 = '|'.join(l6)


# In[9]:


import numpy as np
vec1['child'] = np.where(vec1['CONTENT'].str.contains(regstr1),1,0)
vec1['covid'] = np.where(vec1['CONTENT'].str.contains(regstr2),1,0)
vec1['wait'] = np.where(vec1['CONTENT'].str.contains(regstr3),1,0)
vec1['money'] = np.where(vec1['CONTENT'].str.contains(regstr4),1,0)

vec1['answer'] = np.where(vec1['CONTENT'].str.contains(regstr5),1,0)
vec1['bad'] = np.where(vec1['CONTENT'].str.contains(regstr6),1,0)


# In[10]:


vec1.head(3)


# In[11]:


import numpy as np
X = vec1[['child','covid','wait','money','answer','bad']]
y = vec1['label']
XX = np.array(X)
yy = np.array(y)


# In[12]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2,random_state=109) # 70% training and 30% test


# In[15]:


model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[16]:


me.shape


# In[17]:


me = pd.DataFrame() 
me['pred'] = y_pred
me['act'] = y_test


# In[18]:


data_crosstab = pd.crosstab(me['act'], 
                            me['pred'],
                                margins = False)
print(data_crosstab)


# In[19]:


from sklearn.metrics import precision_recall_fscore_support as score

predicted = me['pred'].tolist()
y_test = me['act'].tolist()

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[20]:


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

