# IR query preprocessor

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import pickle
import json
# import gzip
import os
import subprocess
# import numpy as np
# import multiprocessing
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
# import utils



class Query:
    def __init__(self, ir_toolkit_location, query_file, query_parameter_file, run_filename, stopwords_file):
        self.ir_toolkit_location = ir_toolkit_location
        self.query_file = query_file
        self.query_parameter_file = query_parameter_file
        self.run_filename = run_filename
        self.stopwords_file = stopwords_file
        
#     def build(self, ir_tool_params):
    def run(self):
        
#         utils.create_dir(self.index_location)
    
        query_command = self.ir_toolkit_location + 'runquery/IndriRunQuery'
        toolkit_parameters = [
                                query_command,
                                self.query_file,
                                self.query_parameter_file,
                                self.stopwords_file]
        print(toolkit_parameters)
        with open(self.run_filename, 'wt') as rf:
#             proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            proc2 = subprocess.Popen(['grep', '^.*[ ]Q0[ ]'],stdin=proc.stdout, stdout=rf, stderr=subprocess.STDOUT, shell=False)
            (out, err)= proc2.communicate()

#             print('Run error: ', err)
            if err == None:
                pass
#                 return 'Ok'

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
        

if __name__ == "__main__":
    
    
# #     ir_toolkit_location = sys.argv[1] # '../indri/'

#     # create dataset files dir
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
#     parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--data_split',   type=str, help='')
    parser.add_argument('--pool_size',   type=int, help='')
#     parser.add_argument('--build_index', action='store_true')
#     parser.add_argument('--fold', type=str,   help='')
#     parser.add_argument('--gen_features', action='store_true')
    
    
    args=parser.parse_args()
#     args = fakeParser()
    ir_toolkit_location = '../indri-l2r/'
    
    workdir = './workdir/'
    confdir = './tvqa_config/'

    to_index_input = workdir + 'index_inputs/'
    index_dir = workdir + 'index_dir/'
    parameter_file_location = confdir + '_index_param_file'
    stopwords_file = confdir + 'stopwords'
    all_sub_files = workdir + 'sub_files/'
    all_retrieved_files = workdir + 'retrieved_files/'
    

    if args.data_split == 'all':
        data_splits = ['train', 'dev', 'test']
    else:
        data_splits = [args.data_split]


    for data_split in data_splits:
        
        subtitles_topics_file = all_sub_files + 'subtitle_indri_query_file_' + data_split

         # Run query

        run_filename = all_retrieved_files + 'run_tfidf_' + data_split
        query_parameter_file = confdir + 'tvqa_query_params'


        tfidf_query = Query(ir_toolkit_location, subtitles_topics_file, query_parameter_file, run_filename, stopwords_file)
        tfidf_query.run() # fast
