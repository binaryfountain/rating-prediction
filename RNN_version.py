#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
dt = pd.read_excel('/home/murali/Downloads/Rating_1_and_2.xlsx')


# In[14]:


dt.head(2)
# dt.columns=['DOCUMENT_ID','CONTENT_y','CONTENT_x']


# In[15]:


d12 = dt[dt['RATING'].isin(['Overall Rating 2','Overall Rating 1'])]


# In[16]:


d12 = d12[['CONTENT','RATING']]
d12.columns = ['Copy','Label']


# In[17]:


import numpy as np
d12['label'] = np.where(d12['Label'] == 'Overall Rating 1',1,0)
d12.drop(['Label'],axis=1,inplace=True)
d12.rename({'label':'Label'},axis=1,inplace=True)


# In[18]:


train = d12.sample(frac=0.8, random_state=25)
test = d12.drop(training_data.index)


# In[22]:


train.to_csv('/home/murali/Documents/rating_12_lstmProject/lstm_train/train.csv',index=False)
test.to_csv('/home/murali/Documents/rating_12_lstmProject/lstm_test/test.csv',index=False)


# In[23]:


from nltk.corpus.reader.chasen import test
import numpy as np
import pandas as pd
import os
import os
import nltk
import re
import numpy as np
from os import listdir
import pickle
from os.path import isfile, join
import tensorflow
from tensorflow import keras 
import joblib
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input, Lambda, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import string
from nltk.probability import FreqDist
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical


# In[24]:


stop_words = set(stopwords.words( 'english' ))

# dir_path = os.path.dirname(os.path.realpath(__file__))
HOME_DIR = os.path.expanduser('~')


# In[25]:


sentiment_path = HOME_DIR+'/Documents/rating_12_lstmProject/'


# In[26]:


#sentiment_path = HOME_DIR+"/WholeCommentTaggingProject/"
model_path = "/users/admin/tagging_model/"

model_creation_name = "lstm"


# In[27]:


def clean_doc(doc):
    doc = doc.lower()
    doc = re.sub(r'http\S+', '', doc)
    doc = re.sub(r'http', '', doc)
    doc = re.sub(r"@\S+", "",doc)
    doc = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ",doc)
    doc = re.sub(r"@", "at",doc)

    tokens = doc.split()
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub( '' , w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    doc = ' '.join(tokens)
    return doc

def get_data(data_path):

    file_path = sentiment_path+model_creation_name+"_"+data_path
    print(file_path)
    train = pd.read_csv(file_path+"/"+data_path+".csv",sep=",")
    train = train.dropna(subset=["Copy","Label"])
    train = train.astype({"Copy": str, "Label": int})
    print(train["Label"].value_counts())

    train_1 = train[train["Label"] == 1]
    train_0 = train[train["Label"] == 0]
    train = pd.concat([train_0,train_1])

    return train


# In[28]:


def filter_based_on_vocab(comment,vocab_list):
    tokens = comment.split()
    tokens = [w for w in tokens if w in vocab_list]
    comment = ' '.join(tokens)
    return comment

def create_tokenizer(documents):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(documents)
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding= 'post' )
    return padded

def define_lstm_model(voc_size,max_length):
    model = Sequential()
    model.add(Embedding(voc_size,300,input_length=max_length))
    model.add(Bidirectional(LSTM(200, dropout = 0.2, recurrent_dropout = 0.0)))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model

# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation= 'relu' ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    model.summary()
    #plot_model(model, to_file= 'model.png' , show_shapes=True)
    return model


# In[29]:


def train_model():

    train_data = get_data("train")
    test_data = get_data("test")

    train_data["Copy"] = train_data["Copy"].apply(clean_doc)
    test_data["Copy"] = test_data["Copy"].apply(clean_doc)

    ##### creating vocab list #########
    v= pd.Series(' '.join(train_data['Copy']).split()).value_counts()
    v = pd.DataFrame(v)
    v = v[v[0] > 2]
    v.reset_index(inplace=True)
    v1 = v.drop(0,axis=1)
    v1.to_csv(sentiment_path+model_creation_name+'_model/vocab.csv',index=False)
    v1 = pd.read_csv(sentiment_path+model_creation_name+'_model/vocab.csv')
    vocab_list = set(list(v1['index']))

    print("vocab list length = "+str(len(vocab_list)))

    train_data["Copy"] = train_data["Copy"].apply(lambda x: " ".join([t for t in x.split() if t in vocab_list]))
    test_data["Copy"] = test_data["Copy"].apply(lambda x: " ".join([t for t in x.split() if t in vocab_list]))
    
    tokenizer = create_tokenizer(train_data["Copy"])

    try:
        with open(sentiment_path+model_creation_name+'_model/custom_tokenzier.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)

    
    try:
        with open(sentiment_path+model_creation_name+'_model/custom_tokenzier.pickle', 'rb') as handle:
            tokenizer1 = pickle.load(handle)
    except Exception as e:
        print(e)

    vocab_size = len(tokenizer.word_index) + 1
    max_length = 300

    Xtrain = encode_docs(tokenizer, max_length, train_data["Copy"].tolist())
    Xtest = encode_docs(tokenizer, max_length, test_data["Copy"].tolist())

    y_train = train_data["Label"].tolist()
    y_test = test_data["Label"].tolist()

    model = define_lstm_model(vocab_size,max_length)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    #model = define_model(vocab_size, max_length)
    model.fit(np.array(Xtrain), np.array(y_train), epochs=2, verbose=1, validation_data=(np.array(Xtest),np.array(y_test)))
    model.save(sentiment_path+model_creation_name+'_model/'+model_creation_name+'_model.h5' )

    preds = model.predict(Xtest)
    y_pred = []
    for rec in preds:
        pred_label = np.argmax(rec)
        y_pred.append(pred_label)

    y_true = test_data["Label"].tolist()
    print(classification_report(y_true,y_pred))

def test_model():

    max_length = 300
    test_data = get_data("test")

    test_data["Copy"] = test_data["Copy"].apply(clean_doc)

    v1 = pd.read_csv(sentiment_path+model_creation_name+'_model/vocab.csv')
    vocab_list = set(list(v1['index']))
    test_data["Copy"] = test_data["Copy"].apply(lambda x: " ".join([t for t in x.split() if t in vocab_list]))

    tokenizer = None
    try:
        with open(sentiment_path+model_creation_name+'_model/custom_tokenzier.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        print(e)

    vocab_size = len(tokenizer.word_index) + 1
    Xtest = encode_docs(tokenizer, max_length, test_data["Copy"].tolist())
    y_true = test_data["Label"].tolist()

    model = define_lstm_model(vocab_size,max_length)
    model.load_weights(sentiment_path+model_creation_name+'_model/'+model_creation_name+'_model.h5')
    
    preds = model.predict(Xtest)
    y_pred = []
    for rec in preds:
        pred_label = np.argmax(rec)
        y_pred.append(pred_label)

    print(classification_report(y_true,y_pred))


# In[30]:


train_model()


# In[31]:


test_model()

