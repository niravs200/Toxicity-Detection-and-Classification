#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
df = pd.read_csv("D:/Semester 2/Natural Language Processing/train.csv", sep = ",")
df["comment_text"].head()


# In[4]:


df["comment_text"]= df["comment_text"].str.lower()


# In[5]:


import re
#df["comment_text"] = re.sub(r"\d+","", df["comment_text"])
df["comment_text"] = df["comment_text"].str.replace(r"\d+","")
#df["comment_text"].head()
print(df)


# In[6]:


df["comment_text"] = df["comment_text"].str.replace(r"[^a-z]+"," ")
df["comment_text"].head()


# In[7]:


df["comment_text"] = df["comment_text"].str.strip()
df["comment_text"].head()


# In[1]:


import nltk
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))


# In[10]:


df["commentText_noStopwords"] = df["comment_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopset)]))
df["commentText_noStopwords"].head()


# In[11]:


df["tokenized_comment_text"] = df.apply(lambda row: nltk.word_tokenize(row["commentText_noStopwords"]), axis=1)
df["tokenized_comment_text"].head()


# In[ ]:




