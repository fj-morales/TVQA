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


def build_all_indexes(q_data):
    query_id = q_data['qid']
    query = q_data['q']
    sub = q_data['located_sub_text']
    answers = [q_data['a' + str(i)] for i in range(0,5)]
    
#     print(query_id)
    
    index_input_dir = all_index_inputs + str(query_id) + '_input/' 
    index_input_file = index_input_dir + str(query_id) + '_trec_input_file'
    
    index_location = all_index_dir + str(query_id) + '_index' 
    
    trec_query_file = all_query_files + str(query_id) + 'trec_query_file'
    
    trec_sub_file = all_sub_files + str(query_id) +  'trec_sub_file'
    
    
    if os.path.exists(index_input_dir):
        shutil.rmtree(index_input_dir)
        os.makedirs(index_input_dir)
    else:
        os.makedirs(index_input_dir)
        
    if os.path.exists(index_location):
        shutil.rmtree(index_location)
        os.makedirs(index_location)
    else:
        os.makedirs(index_location)
    
    # generate index input file
    trec_answers = answers_to_trec(answers)    
    to_trecfile(trec_answers, index_input_file, compression = 'no')
    
    # build index
    build_index(index_input_dir, index_location)
    
    # generate query file
    trec_query = query_to_trec(query_id, query)
    to_trecfile(trec_query, trec_query_file, compression = 'no')
    
    
    # generate subtitle query file
    trec_sub = query_to_trec(query_id, sub)
    to_trecfile(trec_sub, trec_sub_file, compression = 'no')

# In[ ]:


# In[6]:


def call_build_index(questions_data):
    # Multiprocessing stuff

    make_folder(all_index_dir)
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


def query_to_trec(q_id, query):
    q_t = {}
    q_t[q_id] = '<top>\n\n' +         '<num> Number: ' + str(q_id) + '\n' +         '<title> ' + query + '\n\n' +         '<desc> Description:' + '\n\n' +         '<narr> Narrative:' + '\n\n' +         '</top>\n\n'
    return q_t
    
#     queries_list = []
#     queries_dict = {}
#     query = {}
#     id_num = 0
#     ids_dict = {}
#     q_trec = {}
#     for query in q_dup_pos:
#         str_id = str(id_num)
#         id_new = str_id.rjust(15, '0')
        
#         key = query['doc_id']
#         q = questions[key]
# #         print(key)
#         text = remove_sc(q['title'] + ' ' + q['text']) #Join title and text 
#         query['number'] = key
# #         query['text'] = '#stopword(' + text + ')'
#         query['text'] = '(' + text + ')'
#         queries_list.append(dict(query))
        
#         q_t = '<top>\n\n' +           '<num> Number: ' + id_new + '\n' +           '<title> ' + text + '\n\n' +           '<desc> Description:' + '\n\n' +           '<narr> Narrative:' + '\n\n' +           '</top>\n\n'
#         q_trec[key] = q_t
# #         print(q)
#         ids_dict[str(id_num)] = key
#         id_num += 1
        
#     queries_dict['queries'] = queries_list
#     # with open(filename, 'wt', encoding='utf-8') as q_file:
#     with open(filename, 'wt') as q_file: #encoding option not working on python 2.7
#         json.dump(queries_dict, q_file, indent = 4)
        
#     return [q_trec, ids_dict]
        
#         ########################
#         ########################


# In[ ]:


# In[8]:


def retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, hits, b=0.2, k=0.8, N=10, M=10, Lambda=0.5):
#     print(q_topics_file)
    #print(hits)
    anserini_search = anserini_loc + 'target/appassembler/bin/SearchCollection'
    command = [ 
               'sh',
               anserini_search,
               '-topicreader',
                'Trec',
                '-index',
                index_loc,
                '-topics',
                q_topics_file,
                '-output',
                retrieved_docs_file,
                '-bm25',
                '-b',
                str(b),
                '-k1',
                str(k),
                '-rm3',
                '-rm3.fbDocs',
                str(int(N)),
                '-rm3.fbTerms',
                str(int(M)),
                '-rm3.originalQueryWeight',
                str(Lambda),
                '-hits',
                str(hits), 
                '-threads',
                '10'
               ]
#     print(command)
#     command = command.encode('utf-8')
    anserini_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    (out, err) = anserini_exec.communicate()
#     print(out)
#     print('Searching error: ', err)


# In[ ]:


# In[9]:


def generate_preds_file(retrieved_docs_file):
    
    with open(retrieved_docs_file, 'rt') as f_in:
        try: 
            for doc in f_in:
#                 print(doc)
                q_id = doc.split(' ')[0]
#                 print(doc.split(' ')[2])
                pred_ans_id = doc.split(' ')[2]

            return pred_ans_id
        except:
            pred_ans_id = int(6) # When BM25+RM3 does not find any document, return an answer index outside the valid answer range
            return pred_ans_id
    


# In[10]:


def baseline_compute(q_data, temp_dir, b,k,N,M,Lambda):
    query_id = q_data[0]
    query = q_data[1] # either question, subtitle, retrieval model?
    answers = q_data[2] # answers, always the same set
#     print(query_id)
    
#     temp_dir = workdir + str(query_id) + '_temp'  + '/'
#     temp_index_input_dir = temp_dir + 'input_to_index/'
#     temp_index_dir = temp_dir + 'index/'
#     temp_index_input_file = temp_index_input_dir + 'trec_doc_input_file'
#     temp_trec_query_file = temp_dir + 'trec_query_file'
    index_location = all_index_dir + str(query_id) + '_index' 
    trec_query_file = all_query_files + str(query_id) + 'trec_query_file'
    retrieved_doc_file = all_retrieved_files + temp_dir + str(query_id) + '_retrieved_doc_file'
    
    
#     if os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)
#         os.makedirs(temp_dir)
#         os.makedirs(temp_index_input_dir)
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)
#         os.makedirs(temp_index_input_dir)
    
#     # generate index input file
#     trec_answers = answers_to_trec(answers)
#     to_trecfile(trec_answers, temp_index_input_file, compression = 'no')
    
#     # build index
#     build_index(temp_index_input_dir, temp_index_dir)
    
#     # generate query file
#     trec_query = query_to_trec(query_id, query, temp_trec_query_file)
#     to_trecfile(trec_query, temp_trec_query_file, compression = 'no')
    
    # get baseline scores file
    # hits = 1, because we are interested in the closest answer, nothing else
    retrieve_docs(trec_query_file, retrieved_doc_file, index_location, hits, b, k, N, M, Lambda)
    
    # Generate predictions
    predicted_answer_id = generate_preds_file(retrieved_doc_file)

    # 
####     shutil.rmtree(temp_index_input_dir)
#     shutil.rmtree(temp_dir)
    
    return predicted_answer_id


# In[ ]:


# In[11]:


def evaluate(predicted_answers, gold_answers):
    
    preds = np.asarray(predicted_answers)
    targets = np.asarray(gold_answers)
    acc = sum(preds == targets) / float(len(preds))
    return acc


# In[12]:


def evaluate_params(params):
    b = params[0]
    k = params[1]
    N = params[2]
    M = params[3]
    Lambda = params[4]
    temp_dir = uuid.uuid4().hex
    
    pred_answers = []
    gold_answers = []
    for item in questions_data:
        answers = [item['a' + str(i)] for i in range(0,5)]
        if model_type == 'qa':
            q_data = [item['qid'], item['q'], answers]
        elif model_type == 'sa':
            q_data = [item['qid'], item['q'], answers]
        elif model_type == 'retrieval':
            print('Retrieval model to be built...')
#             q_data = [item['qid'], item['q'], answers]

        predicted_answer_id = baseline_compute(q_data, temp_dir, b,k,N,M,Lambda)
        pred_answers.append(int(predicted_answer_id))
        gold_answers.append(int(item['answer_idx']))
    
    acc = evaluate(pred_answers, gold_answers)
    results = [
        b,
        k,
        N,
        M,
        Lambda,
        float(acc)
    ]
    
    return results


# In[13]:


def start_process():
    print( 'Starting', multiprocessing.current_process().name)


# In[14]:


def get_random_params(hyper_params, num_iter):
    random_h_params_list = []
    while len(random_h_params_list) < num_iter:
        random_h_params_set = []
        for h_param_list in hyper_params:
            sampled_h_param = random.sample(list(h_param_list), k=1)
#             print(type(sampled_h_param[0]))
#             print(sampled_h_param[0])
            random_h_params_set.append(round(sampled_h_param[0], 3))
        if not random_h_params_set in random_h_params_list:
            random_h_params_list.append(random_h_params_set)
#             print('Non repeated')
        else:
            print('repeated')
    return random_h_params_list


# In[ ]:


# In[19]:


def find_best_dev_model(best_model_params_file, n_rand_iter, pool_size):
#     random_search = 'yes'
    make_folder(all_retrieved_files)
    if random_search == 'yes':
        ## Heavy random search
        brange = np.arange(0.1,1,0.05)
        krange = np.arange(0.1,4,0.1)
        N_range = np.arange(5,500,1) # num of docs
        M_range = np.arange(5,500,1) # num of terms
        lamb_range = np.arange(0,1,0.1) # weights of original query

        ## Light random search
#         brange = [0.2]
#         krange = [0.8]
#         N_range = np.arange(1,50,2)
#         M_range = np.arange(1,50,2)
#         lamb_range = np.arange(0,1,0.2)
        
        h_param_ranges = [brange, krange, N_range, M_range, lamb_range]
        params = get_random_params(h_param_ranges, n_rand_iter)

    else:
        brange = [0.2]
        krange = [0.8]
        N_range = [11]
        M_range = [10]
        lamb_range = [0.5]
       
        params = [[round(b,3), round(k,3), round(N,3), round(M,3), round(Lambda,3)] 
                  for b in brange for k in krange for N in N_range for M in M_range for Lambda in lamb_range]
    
#     print(len(params))
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

#     pool_outputs = pool.map(baseline_computing, params)
    

    pool_outputs = pool.map_async(evaluate_params, params)
    print(pool_outputs.get())
    ###

    
    ##
    
    
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
    params_file = './baselines/best_ir_model/' + 'tvqa' + '_' + 'bm25_rm3_' + data_split + '_hparams.pickle'
    pickle.dump(pool_outputs.get(), open(params_file, "wb" ) )
    print('Total parameters tested: ' + str(len(pool_outputs.get())))
    best_model_params = max(pool_outputs.get(), key=lambda x: x[5])
    
    best_model_dict = {
        'b': best_model_params[0],
        'k': best_model_params[1],
        'N': best_model_params[2],
        'M': best_model_params[3],
        'Lambda': best_model_params[4],
        'n_rand_iter': n_rand_iter,
        'hits': hits,
        'auc05_score': best_model_params[5],
    }
    best_model_dict = {k:str(v) for k, v in best_model_dict.items()} # everything to string
    
    print(best_model_dict)
    with open(best_model_params_file, 'wt') as best_model_f:
        json.dump(best_model_dict, best_model_f)



# In[16]:


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


# In[17]:


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

# def main(args):
    start = datetime.datetime.now()
    print(start)
#     dev_file = 
    test_file = './data/tvqa_val_processed.json' # Val is being used as test!
    build_index_flag = 'yes'
    random_search = 'yes'
    workdir = './workdir/'
    
    hits = 1
    
    anserini_loc = '../anserini/'
#     n_rand_iter = 5000
    
    
    dev_file = sys.args[1] # './data/tvqa_new_dev_processed.json' # Original train data was split in new_train, new_dev
    data_split = sys.args[2] # 'dev'
    n_rand_iter = sys.args[3] # 5000
    pool_size = sys.args[4] # 20
    
    
    with open(dev_file, 'r') as f:
        processed_data_dev = json.load(f)

    with open(test_file, 'r') as f:
        processed_data_test = json.load(f)
    
    # Options
    data_split = 'dev'
    model_type = 'qa'
#     model_type = 'sa'
#     model_type = 'retrieval'
    
    
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    
    if data_split == 'dev':
        print('Dev mode: ')
        print('Total elements: ', len(processed_data_dev))
        questions_data = processed_data_dev
	
    	all_index_dir = workdir + 'index_dirs_dev/'
    	all_index_inputs = workdir + 'index_inputs_dev/'
        all_query_files = workdir + 'query_files_dev/'
        all_sub_files = workdir + 'sub_files_dev/'
        all_retrieved_files = workdir + 'retrieved_files_dev/'
        
    elif data_split == 'test':
        print('Test mode: ')
        questions_data = processed_data_test
	
    	all_index_dir = workdir + 'index_dirs_test/'
    	all_index_inputs = workdir + 'index_inputs_test/'
        all_query_files = workdir + 'query_files_test/'
        all_sub_files = workdir + 'sub_files_test/'
        all_retrieved_files = workdir + 'retrieved_files_dev/'
 

    
    
#    questions_data = questions_data[0:10]
    call_build_index(questions_data)

    best_model_params_file = './baselines/best_ir_model/tvqa_bm25_rm3_best_model_dev.json'
    
    find_best_dev_model(best_model_params_file, n_rand_iter, pool_size)
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

