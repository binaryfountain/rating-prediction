#!/usr/bin/env python
# coding: utf-8

# $$  (a+b)^2 = a^2 + b^2 + 2ab $$

# In[2]:


import pandas as pd
dt = pd.read_csv('/home/murali/Desktop/rating_data/consol_v1.csv')

# dt4 = dt[dt['CONTENT']=='Overall Rating 4']
# dt4 = dt[dt['DOCUMENT_ID'].isin([137029819,137030466,137030196,137030843,137031040,
# 137032998,123676421,137051132,136922892,121793572])]
# dt4.to_csv('/home/murali/Desktop/rating_4.csv',index=False)


# In[9]:


dt.head(2)
dt.columns=['DOCUMENT_ID','CONTENT_y','CONTENT_x']


# In[4]:


dt = pd.read_csv('/home/murali/Desktop/Rating_1_and_2.csv')
dt.head(2)


# In[5]:


dt.groupby(['RATING','New Rating']).size()


# In[6]:


dt.columns=['DOCUMENT_ID','rem1','CONTENT_x','CONTENT_y','rem2']
dt = dt[['DOCUMENT_ID','CONTENT_y','CONTENT_x']]


# In[12]:





# In[10]:


# dt['count'] = dt['CONTENT_x'].str.split().str.len()
# print(dt.shape)
# dt = dt[dt['count'] > 50]
# print(dt.shape)
# dt.drop(['count'],axis=1,inplace=True)


# In[5]:


148086/ 429218


# In[8]:


print(dt.shape)
d12 = dt[dt['CONTENT_y'].isin(['2','1'])]
print(d12.shape)


# In[11]:


d12 = dt[dt['CONTENT_y'].isin(['Overall Rating 2','Overall Rating 1'])]
# dt_comm = dt[~dt['CONTENT_y'].isin(['Overall Rating 5','Overall Rating 4','Overall Rating 3','Overall Rating 2','Overall Rating 1'])]


# In[9]:


d12['CONTENT_y'].value_counts()


# In[8]:


# dt_rt = d12[['DOCUMENT_ID','CONTENT_y']]
# dt_comm_1 = dt_comm[['DOCUMENT_ID','CONTENT_x']]
# print(dt_comm_1.shape)
# dt_comm_1.drop_duplicates(keep=False,inplace=True)
# print(dt_comm_1.shape)


# In[9]:


# dt_comm_2 = dt_comm_1.groupby('DOCUMENT_ID').agg({
#                              'CONTENT': ' '.join }).reset_index()


# In[10]:


# d12 = pd.merge(dt_comm_2, dt_rt, how='inner', on=['DOCUMENT_ID'])


# In[11]:


# dt_rt = dt_rt[['DOCUMENT_ID','CONTENT']]


# In[10]:



dt = d12
dt['CONTENT_y'].value_counts()


# In[11]:


dt.head(2)


# In[16]:


# dt.to_csv('/home/murali/Desktop/new_rating_1_2.csv',index=False)


# In[15]:


# dt_rt1 = dt[dt['CONTENT_y'].isin(['Overall Rating 1'])]
# dt_rt11 = dt_rt1.sample(n=5000)
# dt_rt2 = dt[dt['CONTENT_y'].isin(['Overall Rating 2'])]
# dt_rt22 = dt_rt2.sample(n=5000)
# dt_rt = dt_rt11.append(dt_rt22)
# dt_rt.shape


# In[16]:


# d12 = dt_rt


# In[12]:


d12 = d12.sample(frac=1)


# In[13]:


d12['CONTENT_y'] = d12['CONTENT_y'].astype(int)


# In[18]:


# d12.to_csv('/home/murali/Desktop/fifty_new_rating_1_2.csv',index=False)


# In[ ]:





# In[14]:


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


# In[18]:


# d_5.shape


# In[15]:


d12['label'] = d12['CONTENT_y']


# In[16]:


d12.head(5)


# In[20]:


# encoded_dict = {'Overall Rating 1':1,'Overall Rating 2':2,'Overall Rating 3':3,'Overall Rating 4':4,'Overall Rating 5':5}
# d12['label'] = d12.CONTENT_y.map(encoded_dict)


# In[17]:


d12['label'].value_counts()


# In[18]:


d12 = d12[['DOCUMENT_ID','CONTENT_y','CONTENT_x','label']]
d12.head(2)


# In[19]:


d12.columns = ['DOCUMENT_ID','target','excerpt','label']


# In[20]:


d12['excerpt'] = d12['excerpt'].str.lower()


# In[21]:


d11 = d12[d12['label']==1]
d_1 = d11.excerpt.str.split(expand=True).stack().value_counts()
d_1 = pd.DataFrame(d_1)
d_1.reset_index(inplace = True)
d_1.columns = ['index_11','freq_1']

d22 = d12[d12['label']==2]
d_2 = d22.excerpt.str.split(expand=True).stack().value_counts()
d_2 = pd.DataFrame(d_2)
d_2.reset_index(inplace = True)
d_2.columns = ['index_11','freq_2']

d33 = d12[d12['label']==3]
d_3 = d33.excerpt.str.split(expand=True).stack().value_counts()
d_3 = pd.DataFrame(d_3)
d_3.reset_index(inplace = True)
d_3.columns = ['index_11','freq_3']

d44 = d12[d12['label']==4]
d_4 = d44.excerpt.str.split(expand=True).stack().value_counts()
d_4 = pd.DataFrame(d_4)
d_4.reset_index(inplace = True)
d_4.columns = ['index_11','freq_4']

d55 = d12[d12['label']==5]
d_5 = d55.excerpt.str.split(expand=True).stack().value_counts()
d_5 = pd.DataFrame(d_5)
d_5.reset_index(inplace = True)
d_5.columns = ['index_11','freq_5']

print(d_1.shape)
print(d_2.shape)
print(d_3.shape)
print(d_4.shape)
print(d_5.shape)


# In[26]:


# d51 = pd.merge(d_5, d_1, on="index_11",how='left',indicator=True)
# d51 = d51[d51['_merge']=='left_only']
# print(d51.shape)
# d51 = pd.merge(d_5, d_1, on="index_11",how='right',indicator=True)
# d51 = d51[d51['_merge']=='right_only']
# print(d51.shape)


# In[27]:


# d51 = pd.merge(d_5, d_2, on="index_11",how='right',indicator=True)
# d51 = d51[d51['_merge']=='right_only']
# print(d51.shape)
# d51 = d51[['index_11']]
# d51 = pd.merge(d51, d_1, on="index_11",how='left',indicator=True)
# d51 = d51[d51['_merge']=='left_only']
# print(d51.shape)
# print(d51.head(20))


# In[28]:


# d51 = pd.merge(d_5, d_3, on="index_11",how='right',indicator=True)
# d51 = d51[d51['_merge']=='right_only']
# print(d51.shape)
# d51 = d51[['index_11']]
# d51 = pd.merge(d51, d_1, on="index_11",how='left',indicator=True)
# d51 = d51[d51['_merge']=='left_only']
# print(d51.shape)
# print(d51.head(20))


# In[29]:


# d51 = pd.merge(d_5, d_4, on="index_11",how='right',indicator=True)
# d51 = d51[d51['_merge']=='right_only']
# print(d51.shape)
# d51 = d51[['index_11']]
# d51 = pd.merge(d_1, d51, on="index_11",how='right',indicator=True)
# d51 = d51[d51['_merge']=='right_only']
# print(d51.shape)
# print(d51.head(20))


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


# Shuffle your dataset 
shuffle_df = d12.sample(frac=1)

# Define a size for your train set 
train_size = int(0.8 * len(d12))

# Split your dataset 
train = shuffle_df[:train_size]
test = shuffle_df[train_size:]


# In[24]:


train.shape


# In[25]:


import os
import re
#import cmudict


import torch
from transformers import BertTokenizer, BertModel, BertConfig,BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


# In[26]:


device = 'cpu'

model_path = './bert-base-uncased'
tokenizer= BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=1)
model.to(device)
optim = AdamW(model.parameters(), lr=5e-5)


# In[27]:


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


# In[28]:


train_tokens= tokenizer.batch_encode_plus(train['excerpt'].tolist(),max_length=100,padding='longest',truncation=True)
train_dataset = loader(train_tokens, train.label.values)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# In[ ]:





# In[29]:


for epoch in range(2):
    for batch in train_loader:

        print(".", end="", flush=True)        
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).reshape(attention_mask.shape[0],-1)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs[0]
        loss.backward()
        optim.step()


# In[30]:


test_tokens= tokenizer.batch_encode_plus(test['excerpt'].tolist(),max_length=100,padding='longest',truncation=True)
test['target']=0
test_dataset = loader(test_tokens, test.target.values)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# In[31]:


torch.save(model, '/home/murali/Desktop/rating_data/mymodel/model_state_12.pt')
model = torch.load('/home/murali/Desktop/rating_data/mymodel/model_state_12.pt')


# In[32]:


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


# In[33]:


test['target1']= np.concatenate(total_preds, axis=0)


# In[34]:


test.head(5)


# In[35]:


submission= test[['DOCUMENT_ID', 'label','target1']]
submission.to_csv('/home/murali/Desktop/submission_12.csv', index=False)


# In[36]:


# Define the dataset as python lists
# actual   = [136, 120, 138, 155, 149]
# forecast = [134, 124, 132, 141, 149]

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


# In[37]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(actual, forecast))
rms


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


import numpy as np

test['pred'] = np.where(test['target1'] <= 1.3,1,2)


# test['pred'] = np.where(test['target1'] <= 2,1,
#                        np.where((test['target1'] > 2) & (test['target1'] <= 2.5),2,
#                                np.where((test['target1'] > 2.5) & (test['target1'] <= 3),3,
#                                        np.where((test['target1'] > 3) & (test['target1'] <= 3.5), 4,
#                                                np.where(test['target1'] > 3.5,5,6)))))


# In[39]:


test['pred'].value_counts()


# In[40]:


test['pred'].value_counts()


# In[41]:


from sklearn.metrics import precision_recall_fscore_support as score

predicted = test['pred'].tolist()
y_test = test['label'].tolist()

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# In[42]:


test['pred'].value_counts()


# In[43]:


test['label'].value_counts()


# In[44]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, predicted)
cnf_matrix


# In[45]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



print(precision_score(y_test, predicted, average="micro"))
print(recall_score(y_test, predicted, average="micro")) 
print(f1_score(y_test, predicted, average="micro"))
print('/n')
print(precision_score(y_test, predicted, average="macro"))
print(recall_score(y_test, predicted, average="macro"))  
print(f1_score(y_test, predicted, average="macro"))



# In[46]:


print(precision_score(y_test, predicted, average="weighted"))
print(recall_score(y_test, predicted, average="weighted")) 
print(f1_score(y_test, predicted, average="weighted"))


# In[47]:


test.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


d12.to_csv('/home/murali/Desktop/dataset.csv',index=False)


# In[8]:


# d = d12.sample(n=10000)
# d12 = d


# In[9]:


d12['CONTENT_y'].value_counts()


# In[10]:


d12['CONTENT_x'] = d12['CONTENT_x'].str.lower()


# In[11]:


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
d12['CONTENT'] = d12['CONTENT_x'].apply(lambda x : clean_doc(x))


# In[12]:


d12.head(3)


# In[13]:


encoded_dict = {'Overall Rating 1':1,'Overall Rating 2':2,'Overall Rating 3':3,'Overall Rating 4':4,'Overall Rating 5':5}
d12['label'] = d12.CONTENT_y.map(encoded_dict)


# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(d12.CONTENT.values, 
                                                  d12.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=d12.label.values)


# In[15]:


x_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
train = pd.concat([x_train,y_train],axis=1)
train.columns = ['excerpt','target']

x_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(y_val)
test = pd.concat([x_val,y_val],axis=1)
test.columns = ['excerpt','target']

import numpy as np
train['id'] = np.arange(len(train))
test['id'] = np.arange(len(test))


# In[16]:


import copy 
train_data = train.copy(deep=True)
test_data = test.copy(deep=True)


# In[17]:


import os
MODEL_OUT_DIR = './models/bert_regressor'
## Model Configurations
MAX_LEN_TRAIN = 100
MAX_LEN_VALID = 100
MAX_LEN_TEST = 100
BATCH_SIZE = 16
LR = 1e-3
NUM_EPOCHS = 2
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'

if not os.path.isdir(MODEL_OUT_DIR):
    os.makedirs(MODEL_OUT_DIR)


# In[18]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange


# In[19]:


class Excerpt_Dataset(Dataset):

    def __init__(self, data, maxlen, tokenizer): 
        #Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):    
        #Select the sentence and label at the specified index in the data frame
        excerpt = self.df.loc[index, 'excerpt']
        try:
            target = self.df.loc[index, 'target']
        except:
            target = 0.0
        identifier = self.df.loc[index, 'id']
        #Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(excerpt) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
        #Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        input_ids = torch.tensor(input_ids) 
        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        
        target = torch.tensor(target, dtype=torch.float32)
        
        return input_ids, attention_mask, target


# In[20]:


class BertRegresser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128,1)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        return output


# In[21]:


def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
            optimizer.zero_grad()  
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Training loss is {train_loss/len(train_loader)}")
        val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Loss : {}".format(epoch, val_loss))


# In[22]:


def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            
            mean_loss += criterion(output, target.type_as(output)).item()
#             mean_err += get_rmse(output, target)
            count += 1
            
    return mean_loss/count


# In[23]:


def get_rmse(output, target):
    err = torch.sqrt(metrics.mean_squared_error(target, output))
    return err


# In[24]:


def predict(model, dataloader, device):
    predicted_label = []
    actual_label = []
    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
                        
            predicted_label += output
            actual_label += target
            
    return predicted_label


# In[25]:


## Configuration loaded from AutoConfig 
config = AutoConfig.from_pretrained(MODEL_NAME)
## Tokenizer loaded from AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
## Creating the model from the desired transformer model
model = BertRegresser.from_pretrained(MODEL_NAME, config=config)
## GPU or CPU
device = "cpu"
## Putting model to device
model = model.to(device)
## Takes as the input the logits of the positive class and computes the binary cross-entropy 
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()
## Optimizer
optimizer = optim.Adam(params=model.parameters(), lr=LR)


# In[ ]:





# In[26]:


## Training Dataset
train_set = Excerpt_Dataset(data=train_data, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer)
# valid_set = Excerpt_Dataset(data=validation, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)
test_set = Excerpt_Dataset(data=test_data, maxlen=MAX_LEN_TEST, tokenizer=tokenizer)


## Data Loaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
# valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)


# In[27]:


train(model=model, 
      criterion=criterion,
      optimizer=optimizer, 
      train_loader=train_loader,
      val_loader=test_loader,
      epochs = 2,
     device = device)


# In[28]:


output = predict(model, test_loader, device)
output[0].shape
output[0]
out2 = []
for out in output:
    out2.append(out.cpu().detach().numpy())
    
out = np.array(out2).reshape(len(out2))
submission = pd.DataFrame({'id': test['id'], 'target':out})
submission.to_csv('/home/murali/Desktop/submission1.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


from tensorflow.keras.utils import to_categorical


# In[15]:


y_train = to_categorical(train.label)
y_test = to_categorical(test.label)


# In[16]:


# from transformers import BertModel
# bert = BertModel.from_pretrained('bert-base-cased') # good to go


# In[17]:


from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')


# In[ ]:





# In[18]:


# Tokenize the input (takes some time) 
# here tokenizer using from bert-base-cased
x_train = tokenizer(
    text=train.text.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
x_test = tokenizer(
    text=test.text.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


# In[19]:


input_ids = x_train['input_ids']
attention_mask = x_train['attention_mask']


# In[20]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense


# In[21]:


max_len = 70
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids,attention_mask = input_mask)[0] 
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(6,activation = 'sigmoid')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True


# In[22]:


optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)


# In[23]:


train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    validation_data = (
    {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
    ),
  epochs=1,
    batch_size=16
)


# In[26]:


predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
predicted_raw[0]


# In[35]:


predicted_raw


# In[28]:


import numpy as np
y_predicted = np.argmax(predicted_raw, axis = 1)
y_true = test.label


# In[33]:


y_predicted


# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_predicted))


# In[ ]:


# 26.47%
# 3.73%
# 2.53%
# 4.80%
# 62.47%


# In[ ]:


texts = input(str('input the text'))
x_val = tokenizer(
    text=texts,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 
validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
for key , value in zip(encoded_dict.keys(),validation[0]):
    print(key,value)

