#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import bz2
import pandas as pd
# import dbmanager  as dbmanager
from os.path import join
import json
import time 
# from nltk.tokenize import RegexpTokenizer
# import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# nltk.download('punkt') # for english sentences tokenization

# tokenizer = RegexpTokenizer(r'\w+')


# In[ ]:


## Load non processed data
# file = './data/tvqa_qa_release/tvqa_train.jsonl'
# with open(file, 'r') as f:
#     lines = []
#     for l in f.readlines():
#         loaded_l = json.loads(l.strip("\n"))
#         lines.append(loaded_l)


# In[ ]:


help(clean_str)


# In[ ]:


train_file = './data/tvqa_train_processed.json'
val_file = './data/tvqa_val_processed.json'
with open(val_file, 'r') as f:
    processed_data_val = json.load(f)

with open(train_file, 'r') as f:
    processed_data_train = json.load(f)


# In[ ]:


def qa_prediction(data):
    gold_answers = []
    predicted_answers = []
    print(len(data))
    for item in data:
        tfidf_vectorizer = TfidfVectorizer()
        q = [item['q']]
        answers = [item['a' + str(i)] for i in range(0,5)]
    #     print(answers)
        tfidf_q = tfidf_vectorizer.fit_transform(q)
    #     print(tfidf_q)
        tfidf_answers = tfidf_vectorizer.transform(answers)
    #     print(tfidf_answers)
        cosine_similarities = linear_kernel(tfidf_q, tfidf_answers).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        gold_answers.append(item['answer_idx'])
        predicted_answers.append(related_docs_indices[0])
    return [predicted_answers, gold_answers]


# In[ ]:


[predicted_answers, gold_answers] = qa_prediction(processed_data_val)
preds = np.asarray(predicted_answers)
targets = np.asarray(gold_answers)
acc = sum(preds == targets) / float(len(preds))


# In[ ]:


acc


# In[17]:


def sa_prediction(data):
    tfidf_vectorizer = TfidfVectorizer()
    gold_answers = []
    predicted_answers = []
    for item in data:
    #     print(item['q'])
        sub = [item['located_sub_text']]
        answers = [item['a' + str(i)] for i in range(0,5)]
    #     print(answers)
        tfidf_sub = tfidf_vectorizer.fit_transform(sub)
    #     print(tfidf_q)
        tfidf_answers = tfidf_vectorizer.transform(answers)
    #     print(tfidf_answers)
        cosine_similarities = linear_kernel(tfidf_sub, tfidf_answers).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        gold_answers.append(item['answer_idx'])
        predicted_answers.append(related_docs_indices[0])
    return [predicted_answers, gold_answers]


# In[18]:


[predicted_answers, gold_answers] = sa_prediction(processed_data_val)
preds = np.asarray(predicted_answers)
targets = np.asarray(gold_answers)
acc = sum(preds == targets) / float(len(preds))
acc


# In[ ]:


def retrieval_model(val_data, train_data):
    tfidf_vectorizer_q_train = TfidfVectorizer()
    gold_answers = []
    predicted_answers = []
    questions_train = [ele['q'] for ele in train_data]
    tfidf_q_train = tfidf_vectorizer_q_train.fit_transform(questions_train)
#     print(tfidf_q_train)
    j = 0
    for item in val_data:
        j += 1
#         print(j)
    #     print(item['q'])
#         print('main: ', tfidf_q_train.shape)
        try:
            q = [item['q']]
    #         q_train = [q for q in train_data['q']]
    #         answers = [item['a' + str(i)] for i in range(0,5)]
        #     print(answers)
    #         print(q)
            tfidf_q = tfidf_vectorizer_q_train.transform(q)
        #     print(tfidf_q)

        #     print(tfidf_answers)
            cosine_similarities_q_train = linear_kernel(tfidf_q, tfidf_q_train).flatten()
            related_docs_indices_q_train = cosine_similarities_q_train.argsort()[:-5:-1]
            q_similar_idx = related_docs_indices_q_train[0]
    #         print(q_similar)
            gold_a_idx = train_data[q_similar_idx]['answer_idx']
            gold_train_answer = [train_data[q_similar_idx]['a' + str(gold_a_idx)]]
    #         print(gold_train_answer)

            tfidf_vectorizer_val = TfidfVectorizer()
    #         print(gold_train_answer)
            tfidf_q_val = tfidf_vectorizer_val.fit_transform(gold_train_answer)
    #         print('second: ', tfidf_q_val.shape)

            answers = [item['a' + str(i)] for i in range(0,5)]
            tfidf_answers = tfidf_vectorizer_val.transform(answers)
            cosine_similarities = linear_kernel(tfidf_q_val, tfidf_answers).flatten()
            related_docs_indices = cosine_similarities.argsort()[:-5:-1]
            gold_answers.append(item['answer_idx'])
            predicted_answers.append(related_docs_indices[0])
        
        except: 
            print(train_data[q_similar_idx])
        
        if j%1000 == 0:
            print('processed: ', j)
#         gold_answers.append(item['answer_idx'])
#         predicted_answers.append(related_docs_indices[0])
    return [predicted_answers, gold_answers]


# In[ ]:


[predicted_answers, gold_answers] = retrieval_model(processed_data_val, processed_data_train)
preds = np.asarray(predicted_answers)
targets = np.asarray(gold_answers)
acc = sum(preds == targets) / float(len(preds))
acc


# In[ ]:


processed_data_train[3000]['answer_idx']


# In[ ]:


questions_train = [ele['q'] for ele in processed_data_train]
questions_val = [ele['q'] for ele in processed_data_val]


# In[ ]:


questions_train[0:5]


# In[ ]:


td = TfidfVectorizer()
td_2 = TfidfVectorizer()


# In[ ]:


tfidf_q_train = td.fit_transform(questions_train)
tfidf_q_val = td_2.fit_transform(questions_val)


# In[ ]:


tfidf_q_train


# In[ ]:


tfidf_q_val


# In[ ]:


type(td)

