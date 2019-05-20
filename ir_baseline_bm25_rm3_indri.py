#!/usr/bin/env python
# coding: utf-8

# In[26]:


# %load ir_baseline_bm25_rm3_indri.py
#############
# Imports
#############
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

######################
# Function defintions
######################


# In[27]:


def load_json(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)
    

def remove_sc(text):
    text = re.sub(r'<eos>',' ',text) # My method
    text = re.sub(r'[^\w\s]',' ',text) # My method
    return text    


def all_data_load(data_files_list):
    all_data_file = './data/tvqa_all.json'
    if not os.path.exists(all_data_file):
        all_data = []
        [all_data.extend(load_json(x)) for x in data_files_list]

        with open(all_data_file, 'wt') as all_f:
            json.dump(all_data, all_f, indent=4)
        return all_data
    else:
        return load_json(all_data_file)

def doc_to_trec(key, title):
    trec_answer = {}
    doc = '<DOC>\n' +             '<DOCNO>' + str(key) + '</DOCNO>\n' +             '<title>' + title + '</title>\n' +             '</DOC>\n'
    trec_answer[str(key)] = doc
    return trec_answer


# In[28]:


def answers_to_trec(q_data):
    trec_answers = []
    seed = 0
    n_answers = 5
    ids_equivalences = {}
    for item in q_data:
        answer_keys = range(seed,seed+n_answers)
        answers = [item['a' + str(i)] for i in range(0,5)]
        for a_item in zip(answer_keys, answers):
            trec_answer = doc_to_trec(a_item[0], a_item[1])
            trec_answers.append(trec_answer)
            qa_key = str(item['qid']) + '_a' + str(a_item[0]%n_answers)
            ids_equivalences[str(a_item[0])] = qa_key
        seed += n_answers
#     print('equivs len: ', len(ids_equivalences))
    with open(all_data_ids_equiv_file, 'wt') as ids_e_f:
        json.dump(ids_equivalences, ids_e_f, indent=4)
    return [trec_answers, ids_equivalences]
    


# In[29]:


#     trec_answers = {}
#     key = 0
#     for answer in answers:
# #         print(key, '_', answer)
# #         print(answer)
#         doc = '<DOC>\n' +             '<DOCNO>' + str(key) + '</DOCNO>\n' +             '<TITLE>' + answer + '</TITLE>\n' +             '</DOC>\n'
#         trec_answers[str(key)] = doc
#         key += 1


def to_trecfile(docs, filename, compression = 'yes', query=False):
    print('Trying to save file: ', filename)
#     print(len(docs))
    
    if compression == 'yes':
        with gzip.open(filename,'wt') as f_out:
            if query == True:
                f_out.write('<parameters>\n')
#                 f_out.write('<index>\n' + index_loc + '\n</index>\n')
                
            for doc in docs:
                for key, value in doc.items():
                    f_out.write(value)
            if query == True:
                f_out.write('</parameters>\n')
    else:
        with open(filename,'wt') as f_out:
            if query == True:
                f_out.write('<parameters>\n')
#                 f_out.write('<index>\n' + index_loc + '\n</index>\n')
            for doc in docs:
                for key, value in doc.items():
                    f_out.write(value)
            if query == True:
                f_out.write('</parameters>\n')


# In[30]:


def all_data_to_index_input(data_files_list):
    q_data_all = all_data_load(data_files_list)
    index_input_file = all_index_inputs + 'index_input_file_' + data_split
    [trec_answers, ids_equiv] = answers_to_trec(q_data_all) # Use all data instead the data split
    to_trecfile(trec_answers, index_input_file, compression = 'no')


# In[31]:


def invert_ids(inverted_ids_file):
    with open(all_data_ids_equiv_file, 'rt') as ids_equiv_f:
        ids_equiv = json.load(ids_equiv_f)
        inverted_id_equiv = {v: k for k, v in ids_equiv.iteritems()}
    return inverted_id_equiv


# In[32]:


def q_data_to_trec_file(q_data, filename, q_or_s):
    indri_queries = to_indri_queries(q_data, q_or_s)
    to_trecfile(indri_queries, filename, compression = 'no', query = True)
    
def to_indri_query(q_id, query, q_a_ans_aux_ids):
    
    q_t = {}
#     q_text = '<query>\n<type>indri</type>\n' +          '<number>' + str(q_id) + '</number>\n' +         '<text>\n#combine( ' + remove_sc(query) + ')\n</text>\n'
    q_text = '<query>\n<type>indri</type>\n' +          '<number>' + str(q_id) + '</number>\n' +         '<text>\n' + remove_sc(query) + '\n</text>\n'
    for q_a_inv_id in q_a_ans_aux_ids:
        q_text = q_text + '<workingSetDocno>' + q_a_inv_id +'</workingSetDocno>\n'
    q_text = q_text + '</query>\n\n'
    q_t[q_id] = q_text
    
    
    return q_t                    
       


# In[33]:


def to_indri_queries(q_data, q_or_s):
    indri_queries = []
    inverted_ids = invert_ids(inverted_ids_file)

    for item in q_data:
#         print('id: ', item['qid'])
        q_a_ans_aux_ids = []
        for i in range(0,5):
            q_id_ans = str(item['qid']) + '_a' + str(i)
            q_a_ans_aux_ids.append(inverted_ids[q_id_ans])
            
            
#             print('equiv_dict: ', q_id_ans, inverted_ids[q_id_ans])
            
            
        if q_or_s == 'q':
            indri_queries.append(to_indri_query(item['qid'], item['q'], q_a_ans_aux_ids))
        if q_or_s == 's':
            indri_queries.append(to_indri_query(item['qid'], item['located_sub_text'], q_a_ans_aux_ids))
        
        
    return indri_queries


# In[34]:


def make_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)

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

    toolkit_index = toolkit_loc + 'bin/IndriBuildIndex'
    toolkit_parameters = [
#                            'nohup', 
                           'sh',
                           toolkit_index,
                           '-collection',
                           'TrecCollection',
                           '-generator',
                           'JsoupGenerator',
                           '-threads',
                            '16',
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



#     toolkit_parameters = ['ls',
#                           index_loc]


#     print(toolkit_parameters)

    index_proc = subprocess.Popen(toolkit_parameters,
            stdout=subprocess.PIPE, shell=False)
    (out, err) = index_proc.communicate()
#     print(out.decode("utf-8"))
#     print('Index error: ', err)
    if err == 'None':
        return 'Ok'

def retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, hits, b=0.2, k=0.8, N=10, M=10, Lambda=0.5):
    #     print(q_topics_file)
    #print(hits)
    
    index_param = '-index='+ os.path.realpath(index_loc)
#     print(index_param)
    count = '-count=' + str(hits)
    baseline = '-baseline=okapi,b:' + str(b) + ',k1:' + str(k)
    N_val = '-fbDocs=' + str(N)
    M_val = '-fbTerms=' + str(M)
    threads_val = '-threads=' + str(12)
    Lambda_val = '-fbOrigWeight=' + str(Lambda)
    q_topics_file = os.path.realpath(q_topics_file)
    toolkit_search = toolkit_loc + 'bin/IndriRunQuery'
    command = [ 
#                'sh',
               toolkit_search,
               q_topics_file,
               index_param,
               count,
               '-trecFormat=true',
               baseline,
                N_val,
                M_val,
                Lambda_val,
                threads_val
               ]
    print(command)
    #     command = command.encode('utf-8')
    toolkit_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    (out, err) = toolkit_exec.communicate()
    print('Opening file: ', retrieved_docs_file)
    with open(retrieved_docs_file, 'wt') as run_f:
        run_f.write(out.decode("utf-8"))
#     print(out.decode("utf-8"))
    print('Index error: ', err)
    if err == 'None':
        
        return 'Ok'

    
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

    
    


def evaluate(predicted_answers, gold_answers):
    
#     print('preds: ', predicted_answers)
#     print('golds: ', gold_answers)
    
    preds = np.asarray(predicted_answers)
    targets = np.asarray(gold_answers)
    acc = sum(preds == targets) / float(len(preds))
    return acc


# In[12]:


# In[35]:


def load_predictions(retrieved_docs_file):
    preds = []
    golds = []
    
    with open(all_data_ids_equiv_file, 'rt') as ids_equiv_f:
        ids_equiv = json.load(ids_equiv_f)
    
    with open(retrieved_docs_file, 'rt') as r_file:
        
        ret_qs = {}
        for doc in r_file:
            qid_ori = doc.split(' ')[0]
            aux_id = doc.split(' ')[2]
            ret_qs[qid_ori] = aux_id
        
        for qid, gold in gold_answers_dict.items():
#             print('gold_qid: ', qid)
#             print('gold_value: ', gold)
            golds.append(int(gold))
    

            aux_id = ret_qs[qid]
            
            qid_aux = ids_equiv[aux_id].split('_')[0]
            qa_key = ids_equiv[aux_id]
#             print('qid, aux, gold, : ', qid, aux_id, gold, qid_aux)
            if str(qid) == str(qid_aux):
                pred = int(qa_key.split('_')[1].lstrip('a'))
            else:
                pred = int(6) # Found wrong doc qid answer
    
    
    
#             try:
#                 aux_id = ret_qs[qid]
#                 print('qid, aux, gold, : ', qid, aux_id, gold)
#                 qid_aux = ids_equiv[aux_id].split('_')[0]
#                 if str(qid) == str(qid_aux):
#                     pred = int(qa_key.split('_')[1].lstrip('a'))
#                 else:
#                     pred = int(6) # Found wrong doc qid answer
#             except: 
#                 pred = int(7) # Answer not found in index
            preds.append(pred) 
                
#     print('len preds: ', len(preds))
#     print('len golds: ', len(gold_answers_dict))
    
    return [preds, golds]


# In[36]:


def evaluate_params(params):
    b = params[0]
    k = params[1]
    N = params[2]
    M = params[3]
    Lambda = params[4]
    
    params_suffix = 'b' + str(b) + 'k' + str(k) + 'N' + str(N) + 'M' + str(M) + 'Lambda' + str(Lambda) + 'n_rand_iter' + str(n_rand_iter) + 'hits' + str(hits)
    retrieved_docs_file = all_retrieved_files + 'run_bm25_rm3_preds_' + 'tvqa' + '_' + data_split + '_' + params_suffix + '.txt'
    
#     retrieve_docs(q_topics_file, retrieved_docs_file, all_index_dir, hits, b, k, N, M, Lambda)
    retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, hits, b, k, N, M, Lambda)
    
    [pred_answers, gold_answers] = load_predictions(retrieved_docs_file)
    
    temp_id = uuid.uuid4().hex
    temp_dir = all_retrieved_files + temp_id + '/'
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
#     pred_answers = []
#     gold_answers = []
#     for item in questions_data:
#         answers = [item['a' + str(i)] for i in range(0,5)]
#         if model_type == 'qa':
#             q_data = [item['qid'], item['q'], answers]
#         elif model_type == 'sa':
#             q_data = [item['qid'], item['q'], answers]
#         elif model_type == 'retrieval':
#             print('Retrieval model to be built...')
# #             q_data = [item['qid'], item['q'], answers]

#         predicted_answer_id = baseline_compute(q_data, temp_dir, b,k,N,M,Lambda)
#         pred_answers.append(int(predicted_answer_id))
#         gold_answers.append(int(item['answer_idx']))
    
    acc = evaluate(pred_answers, gold_answers)
    results = [
        b,
        k,
        N,
        M,
        Lambda,
        float(acc)
    ]
    temp_file = all_retrieved_files + temp_id + str(int(time.time())) + '.txt'
    string_print = 'task: ' + temp_id + ' finished!\n'
    with open(temp_file, 'wt') as task_file:
        task_file.write(string_print)
        task_file.write(json.dumps(results, indent=4))
    
#     shutil.rmtree(temp_dir)    
    
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
#     print(pool_outputs.get())
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
    params_file = best_model_dir + 'tvqa' + '_' + 'bm25_rm3_dev_hparams.pickle'
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
        'auc_score': best_model_params[5],
    }
    best_model_dict = {k:str(v) for k, v in best_model_dict.items()} # everything to string
    
    print(best_model_dict)
    with open(best_model_params_file, 'wt') as best_model_f:
        json.dump(best_model_dict, best_model_f)



#################
# Main defition
#################


# In[38]:


if __name__ == "__main__":
    random_search = 'yes'
    start = datetime.datetime.now()
    print(start)
    index_loc = '/home/francisco/msc_project/not-a-punching-bag/reproduction/TVQA/workdir5/indri_index'
    
    print('index_loc: ',index_loc)

    train_file = './data/tvqa_train_processed.json'
    val_file = './data/tvqa_val_processed.json'
    test_file = './data/tvqa_test_public_processed.json'
    
    data_files_list = [train_file, val_file, test_file]
    
    input_file = sys.argv[1] # './data/tvqa_new_dev_processed.json'
    data_split = str(sys.argv[2]) # 'dev'
    n_rand_iter = int(sys.argv[3]) # 5000
    pool_size = int(sys.argv[4]) # 20
    model_type = str(sys.argv[5])# q(uestion) / s(ubtitle)
    
    
#     input_file ='./data/tvqa_new_dev_processed.json'
#     data_split = 'dev'
#     n_rand_iter = 1
#     pool_size = 1
#     model_type = 's'
    
    
    build_index_flag = 'yes'
    
    workdir = './workdir5/'
    
    all_data_ids_equiv_file = workdir + 'all_data_ids_equiv.json'
    inverted_ids_file = workdir + 'ids_equiv.json' 
    
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    
    hits = 1
    toolkit_loc = '../indri/'
    
    
#     q_data_all = all_data_load(data_files_list)
    
    q_data = load_json(input_file)
    
    
    #########
    
#    q_data = q_data[0:100]
    
    #########
    
    
    all_index_dir = workdir + 'index_dirs_' + data_split + '/'
    all_index_inputs = workdir + 'index_inputs_' + data_split + '/'
    all_query_files = workdir + 'query_files_' + data_split + '/'
    all_sub_files = workdir + 'sub_files_' + data_split + '/'
    all_retrieved_files = workdir + 'retrieved_files_' + data_split + '/'
    best_model_dir = workdir + 'best_ir_model/'
    
    make_folder(all_index_inputs)
    make_folder(all_query_files)
    make_folder(all_sub_files)
    make_folder(all_index_dir)
    make_folder(best_model_dir)
    
    # Convert answers to one index_inpu_trec_doc_file
    # Save the index numbering equivalences (newid = queryid_a0, queryid_a1, ...)
    
#     index_input_file = all_index_inputs + 'index_input_file_' + data_split
#     [trec_answers, ids_equiv] = answers_to_trec(q_data_all) # Use all data instead the data split
#     to_trecfile(trec_answers, index_input_file, compression = 'no')
#     all_data_to_index_input(data_files_list)
    
    
    # Convert all questions / subtitles to one trec topics file 
    
    query_topics_file = all_query_files + 'query_indri_file_' + data_split
#     q_data_to_trec_file(q_data, query_topics_file, q_or_s = 'q')
    
    subtitles_topics_file = all_query_files + 'subtitle_indri_query_file_' + data_split
    q_data_to_trec_file(q_data, subtitles_topics_file, q_or_s = 's')
    
    # Build index, single process
    ##
    ## build_index(all_index_inputs, all_index_dir)
    
    # Retrieve 1 (most relevant) answer/doc per question - multiprocessing 5000
    
    if model_type == 'q':
        print('Model type: question-answer')
        q_topics_file = query_topics_file
    elif model_type == 's':
        print('Model type: subtitle-answer')
        q_topics_file = subtitles_topics_file
        
    
    # Global gold answers list
    gold_answers_dict = {}
    for q in  q_data:
        gold_answers_dict[str(q['qid'])] = q['answer_idx']

#     print('type gold: ', type(gold_answers_dict.keys()[0]))
    
    # Pick best model according to accuracy
#     if data_split == 'dev':
#         find_best_model()
    best_model_params_file = best_model_dir + 'tvqa' + '_bm25_rm3_best_model_dev.json'
    
#     params = [1,1,1,1,1]
#     evaluate_params(params)
    
    find_best_dev_model(best_model_params_file, n_rand_iter, pool_size)
    
#     print(q_data[0])
    
    # Test on test set
#     if data_split == 'test':
#         evaluate_model()
    

