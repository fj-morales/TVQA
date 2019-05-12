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

def doc_to_trec(key, title):
    trec_answer = {}
    doc = '<DOC>\n' +             '<DOCNO>' + str(key) + '</DOCNO>\n' +             '<TITLE>' + title + '</TITLE>\n' +             '</DOC>\n'
    trec_answer[str(key)] = doc
    return trec_answer

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
            ids_equivalences[qa_key] = a_item[0]
        seed += n_answers
    return [trec_answers, ids_equivalences]
    
#     trec_answers = {}
#     key = 0
#     for answer in answers:
# #         print(key, '_', answer)
# #         print(answer)
#         doc = '<DOC>\n' +             '<DOCNO>' + str(key) + '</DOCNO>\n' +             '<TITLE>' + answer + '</TITLE>\n' +             '</DOC>\n'
#         trec_answers[str(key)] = doc
#         key += 1


def to_trecfile(docs, filename, compression = 'yes'):
    if compression == 'yes':
        with gzip.open(filename,'wt') as f_out:
            for doc in docs:
                for key, value in doc.items():
                    f_out.write(value)
    else:
        with open(filename,'wt') as f_out:
            for doc in docs:
                for key, value in doc.items():
                    f_out.write(value)


def topic_to_trec(q_id, query):
    q_t = {}
    q_t[q_id] = '<top>\n\n' +         '<num> Number: ' + str(q_id) + '\n' +         '<title> ' + query + '\n\n' +         '<desc> Description:' + '\n\n' +         '<narr> Narrative:' + '\n\n' +         '</top>\n\n'
    return q_t                    
                    
                    
def topics_to_trec(q_data, q_or_s):
    trec_topics = []
    
    for item in q_data:
        if q_or_s == 'q':
            trec_topics.append(topic_to_trec(item['qid'], item['q']))
        if q_or_s == 's':
            trec_topics.append(topic_to_trec(item['qid'], item['located_sub_text']))
    return trec_topics


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

                
                
#################
# Main defition
#################

if __name__ == "__main__":

    start = datetime.datetime.now()
    print(start)
    
    input_file = sys.argv[1]
    data_split = str(sys.argv[2]) # 'dev'
    n_rand_iter = int(sys.argv[3]) # 5000
    pool_size = int(sys.argv[4]) # 20

    
    build_index_flag = 'yes'
    
    workdir = './workdir3/'
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    
    hits = 1
    anserini_loc = '../anserini/'
    
    with open(input_file, 'r') as f:
        q_data = json.load(f)
    
#     q_data = q_data[0:100]
    
    all_index_dir = workdir + 'index_dirs_' + data_split + '/'
    all_index_inputs = workdir + 'index_inputs_' + data_split + '/'
    all_query_files = workdir + 'query_files_' + data_split + '/'
    all_sub_files = workdir + 'sub_files_dev' + data_split + '/'
    all_retrieved_files = workdir + 'retrieved_files' + data_split + '/'
    
    make_folder(all_index_inputs)
    make_folder(all_query_files)
    make_folder(all_sub_files)
    make_folder(all_index_dir)
    
    # Convert answers to one index_inpu_trec_doc_file
    # Save the index numbering equivalences (newid = queryid_a0, queryid_a1, ...)
    
    index_input_file = all_index_inputs + 'index_input_file_' + data_split
    [trec_answers, ids_equiv] = answers_to_trec(q_data)
    to_trecfile(trec_answers, index_input_file, compression = 'no')
    
    # Convert all questions / subtitles to one trec topics file 
    trec_queries = topics_to_trec(q_data, q_or_s = 'q')
    
    query_file = all_query_files + 'query_file_' + data_split
    to_trecfile(trec_queries, query_file, compression = 'no')
    
    # Build index, single process
    build_index(all_index_inputs, all_index_dir)
    
    # Retrieve 1 (most relevant) answer/doc per question - multiprocessing 5000
    
    # Pick best model according to accuracy
#     if data_split == 'dev':
#         find_best_model()
    
    
    # Test on test set
#     if data_split == 'test':
#         evaluate_model()
    