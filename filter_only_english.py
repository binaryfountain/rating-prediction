#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
d12 = pd.read_csv('/home/murali/Desktop/delete_after_few_days/right_data_to_qc_rating_1_2.csv')


# In[104]:


import nltk 
nltk.download('words')
words = set(nltk.corpus.words.words())

def removeNonEnglishWordsFunct(x):
    words = set(nltk.corpus.words.words())
    filteredSentence = " ".join(w for w in nltk.wordpunct_tokenize(x)                                 if w.lower() in words or not w.isalpha())
    return filteredSentence


# In[105]:


# d13['new'] = d13['CONTENT'].apply(lambda x : removeNonEnglishWordsFunct(x))


# In[106]:


# d13.to_csv('/home/murali/Desktop/del.csv')


# In[107]:



from spacy_langdetect import LanguageDetector
import spacy
nlp = spacy.load('en_core_web_sm')


# In[108]:


nlp.add_pipe(LanguageDetector(), name='language_detector', last=True) #2
# text_content = "Er lebt mit seinen Eltern und seiner Schwester in Berlin."
d12['new1'] = d12['CONTENT'].apply(lambda x : nlp(x)._.language)
# doc = nlp(text_content) #3
# detect_language = doc._.language #4
# print(detect_language)


# In[109]:


d12['spacy_lang'] = d12['new1'].apply(lambda x: x['language'])
d12['spacy_score'] = d12['new1'].apply(lambda x: x['score'])


# In[110]:


d12['spacy_lang'].value_counts()


# In[111]:


d13 = d12[d12['spacy_lang']=='en']


# In[121]:


d12.shape


# In[122]:


d13.shape


# In[112]:


# d14 = d13


# In[113]:


# ll=list(d14['CONTENT'])
# bb = list()
# import pycld2 as cld2

# for i in range(len(ll)):
#     _, _, _, detected_language = cld2.detect(ll[i],  returnVectors=True)
#     bb.append(detected_language)


# In[114]:


# bbb = pd.DataFrame(bb)
# print(bbb.shape)
# bbb.columns = ['c1','c2','c3','c4']
# bbb['n1'] = bbb['c1'].astype(str)
# bbb['pycld2'] = bbb['n1'].str.split(',').str[2]
# bbb.drop(['c1','c2','c3','c4','n1'],axis=1,inplace=True)
# d14 = pd.concat([d14,bbb],axis=1)
# # d14.head(10)


# In[115]:



# d14['pycld2'].value_counts()


# In[116]:


# d14.head(2)


# In[117]:


# d15 = d14[d14['pycld2'].str.contains('ENGLISH')==False]
# d15.shape


# In[118]:


# d15


# In[119]:


# d14.head(2)


# In[120]:


# from textblob import TextBlob

# d14['new2'] = d14['CONTENT'].apply(lambda x : TextBlob(x).detect_language())


# In[ ]:




