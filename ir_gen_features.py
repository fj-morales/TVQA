# IR query preprocessor

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import pickle
import json
# import gzip
import os
import subprocess
from functools import partial
# import numpy as np
import multiprocessing
# import re 
# import csv
# import torch
import sys
# import shutil
# import random

# import argparse

# import uuid
# import datetime
# import time

# import bz2
import argparse
# import pandas as pd
# # import dbmanager  as dbmanager
# from os.path import join

## My libraries

# import eval_utils
from ir_utils import *

# from utils import *
# from join_split_files import join_files



def features_dict(features_file):
    with open(features_file, 'rt') as f_file:
        feat_dict = {}
        for line in f_file:
            qid = line.split(' ')[1].split(':')[1]
            if qid in feat_dict.keys():
                feat_dict[qid].append(line)
            else:
                feat_dict[qid] = [line]
    return feat_dict

def save_features(all_features_dict,qids_list, features_split_file):
    print(features_split_file)
    with open(features_split_file, 'wt') as out_f:
        for qid in qids_list:
            to_write = all_features_dict[qid]
            out_f.write("".join(to_write))


def generate_features_params(params, feat_param_file):
    with open(params[0], 'rt') as q_trec_f:
        trec_lines = q_trec_f.readlines()
    
    with open(feat_param_file, 'wt') as f_out:
        for line in trec_lines[:-1]:
            f_out.write(line)
        f_out.write('<index>' + params[1] + '</index>\n')    
        f_out.write('<outFile>' + params[2] + '</outFile>\n')    
        f_out.write('<rankedDocsFile>' + params[3] + '</rankedDocsFile>\n')    
        f_out.write('<qrelsFile>' + params[4] + '</qrelsFile>\n')    
        f_out.write('<stemmer>' + params[5] + '</stemmer>\n') # This should not be here!, Fix GenerateExtraFeatures.cpp to read from index manifest
        f_out.write('</parameters>\n')    
         
        
class GenerateExtraFeatures:
    def __init__(self, ir_toolkit_location, feat_param_file):
        self.ir_toolkit_location = ir_toolkit_location
        self.feat_param_file = feat_param_file
        self.log_file = feat_param_file + '_run.log'
        
#     def build(self, ir_tool_params):
    def run(self):
        
        features_command = self.ir_toolkit_location + 'L2R-features/GenerateExtraFeatures'
        toolkit_parameters = [
                                features_command,
                                self.feat_param_file]
        print(toolkit_parameters)
        with open(self.log_file, 'wt') as rf:
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
#             proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            
            (out, err)= proc.communicate()
#             print(out.decode('utf-8'))
            print(err)
        print('Features file generated. Log: ', self.log_file)
            

class fakeParser:
    def __init__(self):
        self.dataset = 'bioasq' 
        self.data_split = 'test'
#         self.data_split = 'train'
#         self.data_split = 'dev'
#         self.build_index = True
        self.build_index = None
        self.fold = '1'
        self.gen_features = True
#         self.gen_features = None
        
def start_process():
    print( 'Starting', multiprocessing.current_process().name)

    
def mini_process(gen_features_param_file):
    feature_generator = GenerateExtraFeatures(ir_toolkit_location, gen_features_param_file)
    feature_generator.run()

if __name__ == "__main__":
        
    
# #     ir_toolkit_location = sys.argv[1] # '../indri/'

#     # create dataset files dir
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
#     parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--data_split',   type=str, help='')
#     parser.add_argument('--fold', type=str,   help='')
    
    args=parser.parse_args()
#     args = fakeParser()
    ir_toolkit_location = '../indri-l2r/'
    
    pool_size = 3
    workdir = './workdir/'
    confdir = './tvqa_config/'
    gen_features_dir = workdir + 'gen_features_dir/'
    
    create_dir(gen_features_dir)
    
    index_dir = workdir + 'index_dir/'
    parameter_file_location = confdir + '_index_param_file'
    stopwords_file = confdir + 'stopwords'
    all_sub_files = workdir + 'sub_files/'
    all_retrieved_files = workdir + 'retrieved_files/'
    
    gold_answer_qrels_file = workdir + 'gold_answer_qrels'
    
    if args.data_split == 'all':
        data_splits = ['train', 'dev', 'test']
    else:
        data_splits = [args.data_split]
    
    feature_param_files = []
    for data_split in data_splits:
        subtitles_topics_file = all_sub_files + 'subtitle_indri_query_file_' + data_split
  
        run_filename = all_retrieved_files + 'run_tfidf_' + data_split
        
        out_features_file = gen_features_dir + 'l2r_features_' + data_split
        gen_features_param_file = workdir + 'gen_features_param_file' + data_split
        feature_param_files.append(gen_features_param_file)
        
        features_params =[
            subtitles_topics_file,
            index_dir,
            out_features_file,
            run_filename,
            gold_answer_qrels_file,
            'none', # This should not be here!, Fix GenerateExtraFeatures.cpp to read from index manifest
        ]    
        
        generate_features_params(features_params, gen_features_param_file)
        

    # Generate L2R features 
    
    pool = multiprocessing.Pool(processes=pool_size,
                            initializer=start_process,
                            )

    pool_outputs = pool.map_async(mini_process, feature_param_files)

    pool.close() # no more tasks

    pool.join()  # wrap up current tasks


