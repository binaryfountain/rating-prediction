#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
dt = pd.read_csv('/home/murali/Desktop/rating_data/social_rating_sep.csv')


# In[109]:


dt.head(2)


# In[110]:


import pandas as pd
# dt = pd.read_csv('/home/murali/Desktop/social22.csv')
d1 = dt[dt['CONTENT'].isin(['Overall Rating 4','Overall Rating 5'])]
dt_comm = dt[~dt['CONTENT'].isin(['Overall Rating 5','Overall Rating 4','Overall Rating 3','Overall Rating 2','Overall Rating 1'])]


# In[111]:


dt.head(4)


# In[112]:


# dt_comm['count'] = dt_comm['CONTENT'].str.split().str.len()
# print(dt_comm.shape)
# dt_comm = dt_comm[dt_comm['count'] > 25]
# print(dt_comm.shape)
# dt_comm.drop(['count'],axis=1,)


# In[113]:


24748 / 192072


# In[114]:


# dt_rt['CONTENT'].value_counts()


# In[115]:


# dt_rt = dt_rt[['DOCUMENT_ID','CONTENT']]
# dt_comm_1 = dt_comm[['DOCUMENT_ID','CONTENT']]
# print(dt_comm_1.shape)
# dt_comm_1.drop_duplicates(keep=False,inplace=True)
# print(dt_comm_1.shape)


# In[116]:


# dt_comm_2 = dt_comm_1.groupby('DOCUMENT_ID').agg({
#                              'CONTENT': ' '.join }).reset_index()


# In[117]:


# d12 = pd.merge(dt_comm_2, dt_rt, how='inner', on=['DOCUMENT_ID'])


# In[118]:


# d1 = dt[dt['SURVEY_QUESTION']=='Overall Rating']
# d1 = d1[d1['CONTENT'].isin(['Overall Rating 4','Overall Rating 5'])]
# d2 = dt[dt['SURVEY_QUESTION']=='Your comments:']
d11 = d1[['DOCUMENT_ID','CONTENT']]
d22 = dt_comm[['DOCUMENT_ID','RESPONSE']]
d22.drop_duplicates(keep=False,inplace=True)

d12 = pd.merge(d11, d22, how='inner', on=['DOCUMENT_ID'])
d12.shape
d12.columns=['DOCUMENT_ID','CONTENT_y','CONTENT_x']


# In[119]:


d11.head(2)


# In[120]:


d22.head(2)


# In[121]:


d12.shape


# In[122]:


d12['CONTENT_y'].value_counts()


# In[123]:


d12.head(2)


# In[124]:


# ll = list(d12['CONTENT_y'].unique())
# ddd = pd.DataFrame()
# for i in range(len(ll)):
#     dd = d12[d12['CONTENT_y']==ll[i]]
#     dd= dd.sample(n=800)
#     ddd = ddd.append(dd)
d12 = d12.sample(n=5000)


# In[125]:


from nltk.corpus import stopwords
import string
import re
# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    doc = doc.lower()
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub( '' , w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    ph = ' '.join(tokens)
    return ph
d12['CONTENT_x'] = d12['CONTENT_x'].apply(lambda x : clean_doc(x))


# In[126]:


import os
import re
#import cmudict


import torch
from transformers import BertTokenizer, BertModel, BertConfig,BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


# In[127]:


class loader(torch.utils.data.Dataset):
    def __init__(self, train_tokens, labels):
        self.train_tokens = train_tokens
        self.labels = labels

    def __getitem__(self, idx):
        
        #train_tokens= tokenizer.batch_encode_plus(self.text_list,max_length=512,padding='longest',truncation=True)
        

        item = {key: torch.tensor(val[idx]) for key, val in self.train_tokens.items()}
        item['labels'] = torch.tensor(self.labels[idx],dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


# In[128]:


encoded_dict = {'Overall Rating 1':1,'Overall Rating 2':2,'Overall Rating 3':3,'Overall Rating 4':4,'Overall Rating 5':5}
d12['label'] = d12.CONTENT_y.map(encoded_dict)


# In[129]:


d12.head(2)


# In[130]:


d12.columns = ['DOCUMENT_ID','target','excerpt','label']


# In[131]:


test=d12


# In[132]:


test.head(2)


# In[133]:


device = 'cpu'

model_path = './bert-base-uncased'
tokenizer= BertTokenizer.from_pretrained(model_path)


# In[134]:


test_tokens= tokenizer.batch_encode_plus(test['excerpt'].tolist(),max_length=100,padding='longest',truncation=True)
test['target']=0
test_dataset = loader(test_tokens, test.target.values)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# In[ ]:





# In[135]:



model = torch.load('/home/murali/Desktop/rating_data/mymodel/model_state_45.pt')
# model = Model()
# model.load_state_dict(torch.load('/home/murali/Desktop/rating_data/mymodel/model_state_all.pth'))


# In[136]:


total_preds = []


# iterate over batches
for step,batch in enumerate(test_loader):
#     print(step)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device).reshape(attention_mask.shape[0],-1)
    with torch.no_grad():
    
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)        
   
        preds = outputs[1].detach().cpu().numpy()
    
        total_preds.append(preds)


# In[137]:


test['target1']= np.concatenate(total_preds, axis=0)


# In[138]:


test.head(10)


# In[139]:


actual = test['label'].tolist()
forecast = test['target1'].tolist()
    
# Consider a list APE to store the
# APE value for each of the records in dataset
APE = []
  
# Iterate over the list values
for day in range(len(actual)):
  
    # Calculate percentage error
    per_err = (actual[day] - forecast[day]) / actual[day]
  
    # Take absolute value of
    # the percentage error (APE)
    per_err = abs(per_err)
  
    # Append it to the APE list
    APE.append(per_err)
  
# Calculate the MAPE
MAPE = sum(APE)/len(APE)
  
# Print the MAPE value and percentage
print(f'''
MAPE   : { round(MAPE, 2) }
MAPE % : { round(MAPE*100, 2) } %
''')


# In[145]:


test.to_csv('/home/murali/Desktop/rating_data/test_results_45.csv', index=False)


# In[155]:


import numpy as np

test['pred'] = np.where(test['target1'] <= 4.2,4,5)


# In[156]:


from sklearn.metrics import precision_recall_fscore_support as score

predicted = test['pred'].tolist()
y_test = test['label'].tolist()

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[157]:


test['pred'].value_counts()


# In[158]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, predicted)
cnf_matrix


# In[159]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



print(precision_score(y_test, predicted, average="micro"))
print(recall_score(y_test, predicted, average="micro")) 
print(f1_score(y_test, predicted, average="micro"))
print('/n')
print(precision_score(y_test, predicted, average="macro"))
print(recall_score(y_test, predicted, average="macro"))  
print(f1_score(y_test, predicted, average="macro"))


# In[90]:


print(precision_score(y_test, predicted, average="weighted"))
print(recall_score(y_test, predicted, average="weighted")) 
print(f1_score(y_test, predicted, average="weighted"))

