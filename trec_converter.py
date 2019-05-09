#!/usr/bin/env python
# coding: utf-8

# In[21]:


# %load ir_baseline_bm25_rm3.py


# ## TVQA - building index separately

# In[ ]:


import pickle
import json
import gzip
import os
import subprocess
import numpy as np
import multiprocessing
import re 
import csv
import torch
import sys
import shutil
import random

import argparse

import uuid
import datetime
import time

import bz2
import pandas as pd
# import dbmanager  as dbmanager
from os.path import join


# from nltk.tokenize import RegexpTokenizer
# import nltk
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


# In[2]:


def answers_to_trec(answers):
    trec_answers = {}
    key = 0
    for answer in answers:
#         print(key, '_', answer)
#         print(answer)
        doc = '<DOC>\n' +             '<DOCNO>' + str(key) + '</DOCNO>\n' +             '<TITLE>' + answer + '</TITLE>\n' +             '</DOC>\n'
        trec_answers[str(key)] = doc
        key += 1
    return trec_answers


# In[ ]:


# def generate_index_dir():

# try:
#                 golden_file = sys.argv[1]
#                 predictions_file = sys.argv[2]
#         except:
#                 sys.exit("Provide golden and predictions files.")
        
#         try:
#                 system_name = sys.argv[3]
#         except :
#                 try:
#                         system_name = predictions_file.split('/')[-1]
#                 except:
#                         system_name = predictions_file

#         with open(golden_file, 'r') as f:
#                 golden_data = json.load(f)

#         with open(predictions_file, 'r') as f:
#                 predictions_data = json.load(f)

#         temp_dir = uuid.uuid4().hex
#         qrels_temp_file = '{0}/{1}'.format(temp_dir, 'qrels.txt')
#         qret_temp_file = '{0}/{1}'.format(temp_dir, 'qret.txt')

#         try:
#                 if not os.path.exists(temp_dir):
#                         os.makedirs(temp_dir)
#                 else:
#                         sys.exit("Possible uuid collision")

#                 format_bioasq2treceval_qrels(golden_data, qrels_temp_file)
#                 format_bioasq2treceval_qret(predictions_data, system_name, qret_temp_file)

#                 trec_evaluate(qrels_temp_file, qret_temp_file)
#         finally:
#                 os.remove(qrels_temp_file)
#                 os.remove(qret_temp_file)
#                 os.rmdir(temp_dir)


# In[ ]:


# In[3]:


def to_trecfile(docs, filename, compression = 'yes'):
    # Pickle to Trectext converter
    doc_list = []
    if compression == 'yes':
        with gzip.open(filename,'wt') as f_out:
            docus = {}
            for key, value in docs.items():
                f_out.write(value)
    else:
        with open(filename,'wt') as f_out:
            docus = {}
            for key, value in docs.items():
                f_out.write(value)


# In[ ]:


# In[4]:


def build_index(index_input, index_loc):
    if build_index_flag == 'no':
        return
# Build corpus index 
    if os.path.exists(index_loc):
        shutil.rmtree(index_loc)
        os.makedirs(index_loc)
    else:
        os.makedirs(index_loc) 
#     index_loc_param = '--indexPath=' + index_loc

    anserini_index = anserini_loc + 'target/appassembler/bin/IndexCollection'
    anserini_parameters = [
#                            'nohup', 
                           'sh',
                           anserini_index,
                           '-collection',
                           'TrecCollection',
                           '-generator',
                           'JsoupGenerator',
                           '-threads',
                            '1',
                            '-input',
                           index_input,
                           '-index',
                           index_loc,
                           '-storePositions',
                            '-keepStopwords',
                            '-storeDocvectors',
                            '-storeRawDocs']
#                           ' >& ',
#                           log_file,
#                            '&']



#     anserini_parameters = ['ls',
#                           index_loc]


#     print(anserini_parameters)

    index_proc = subprocess.Popen(anserini_parameters,
            stdout=subprocess.PIPE, shell=False)
    (out, err) = index_proc.communicate()
#     print(out.decode("utf-8"))
#     print('Index error: ', err)
    if err == 'None':
        return 'Ok'


# In[ ]:


# In[5]:



def query_to_trec(q_id, query):
    q_t = {}
    q_t[q_id] = '<top>\n\n' +         '<num> Number: ' + str(q_id) + '\n' +         '<title> ' + query + '\n\n' +         '<desc> Description:' + '\n\n' +         '<narr> Narrative:' + '\n\n' +         '</top>\n\n'
    return q_t
  

def build_all_indexes(q_data):
    query_id = q_data['qid']
    query = q_data['q']
    sub = q_data['located_sub_text']
    answers = [q_data['a' + str(i)] for i in range(0,5)]
    
#     print(query_id)
    
    index_input_dir = all_index_inputs + str(query_id) + '_input/' 
    index_input_file = index_input_dir + str(query_id) + '_trec_input_file'
    
#     index_location = all_index_dir + str(query_id) + '_index' 
    
    trec_query_file = all_query_files + str(query_id) + 'trec_query_file'
    
    trec_sub_file = all_sub_files + str(query_id) +  'trec_sub_file'
    
    
    if os.path.exists(index_input_dir):
        shutil.rmtree(index_input_dir)
        os.makedirs(index_input_dir)
    else:
        os.makedirs(index_input_dir)
        
    
    # generate index input file
    trec_answers = answers_to_trec(answers)    
    to_trecfile(trec_answers, index_input_file, compression = 'no')
    
    # build index
#     build_index(index_input_dir, index_location)
    
    # generate query file
    trec_query = query_to_trec(query_id, query)
    to_trecfile(trec_query, trec_query_file, compression = 'no')
    
    
    # generate subtitle query file
    trec_sub = query_to_trec(query_id, sub)
    to_trecfile(trec_sub, trec_sub_file, compression = 'no')

# In[ ]:


# In[6]:

def start_process():
    print( 'Starting', multiprocessing.current_process().name)

def call_build_index(questions_data):
    # Multiprocessing stuff

    
    make_folder(all_index_inputs)
    make_folder(all_query_files)
    make_folder(all_sub_files)
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                               )
    pool_outputs = pool.map_async(build_all_indexes, questions_data)

    pool.close() # no more tasks
    while (True):
        if (pool_outputs.ready()): break
        remaining = pool_outputs._number_left
#         remaining2 = remaining1
#         remaining1 = pool_outputs._number_left
        if remaining%10 == 0:
            print("Waiting for", remaining, "tasks to complete...")
            time.sleep(2)
        
      
    pool.join()  # wrap up current tasks
    pool_outputs.get()
        


# In[ ]:


# In[7]:



def make_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)


#############
# MAIN


# In[20]:


if __name__ == "__main__":

    pool_size = 20
    input_file = sys.argv[1] 
    
    
    start = datetime.datetime.now()
    print(start)
    
    workdir = '/ssd/francisco/trec_datasets/tvqa/'
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    if 'new_dev' in input_file:
        print(input_file)
        base_dir = workdir + 'new_dev/'
    elif 'new_train' in input_file:
        print(input_file)
        base_dir = workdir + 'new_train/'
    elif 'val' in input_file:    
        print(input_file)
        base_dir = workdir + 'val/'
    elif 'test' in input_file:    
        print(input_file)
        base_dir = workdir + 'test/'
    
    with open(input_file, 'r') as f:
        processed_data = json.load(f)
    
    questions_data = processed_data
	
    print('Total elements: ', len(processed_data))
    all_index_inputs = base_dir + 'docs/'
    all_query_files = base_dir + 'queries/'
    all_sub_files = base_dir + 'sub_files/'
    
    
#    questions_data = questions_data[0:10]
    call_build_index(questions_data)

#     best_model_params_file = './baselines/best_ir_model/tvqa_bm25_rm3_best_model_dev.json'
    
#     find_best_dev_model(best_model_params_file, n_rand_iter, pool_size)
    end = datetime.datetime.now()
    print(end)


# In[ ]:


# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
#     argparser.add_argument("--cuda", action="store_true")
#     argparser.add_argument("--run_dir",  type=str, default="/D/home/tao/mnt/ASAPPNAS/tao/test")
#     argparser.add_argument("--model", type=str, required=True, help="which model class to use")
#     argparser.add_argument("--embedding", "--emb", type=str, help="path of embedding")
#     argparser.add_argument("--train", type=str, required=True, help="training file")
#     argparser.add_argument("--wasserstein", action="store_true")
#     argparser.add_argument("--cross_train", type=str, required=True, help="cross training file")
#     argparser.add_argument("--eval", type=str, required=True, help="validation file")
#     argparser.add_argument("--batch_size", "--batch", type=int, default=100)
#     argparser.add_argument("--max_epoch", type=int, default=100)
#     argparser.add_argument("--learning", type=str, default='adam')
#     argparser.add_argument("--lr", type=float, default=0.001)
#     argparser.add_argument("--lr2", type=float, default=0.0001)
#     argparser.add_argument("--lambda_d", type=float, default=0.01)
#     argparser.add_argument("--use_content", action="store_true")
#     argparser.add_argument("--eval_use_content", action="store_true")
#     argparser.add_argument("--save_model", type=str, required=False, help="location to save model")

#     args, _  = argparser.parse_known_args()
#     main(args)

