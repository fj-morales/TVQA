# IR_indexing

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import subprocess
import sys

import argparse

## My libraries

import eval_utils
import utils
import bioasq_corpus_parser
import robust_corpus_parser
# import bioasq_query_parser
import query_parser
# from utils import *


# In[ ]:


class Index:
    def __init__(self, ir_toolkit_location, index_dir, parameter_file_location, stopwords_file):
        self.ir_toolkit_location = ir_toolkit_location
        self.parameter_file_location = parameter_file_location
        self.index_dir = index_dir
        self.stopwords_file = stopwords_file
#     def build(self, ir_tool_params):
    def build(self):
        
#         utils.create_dir(self.index_location)
    #     index_loc_param = '--indexPath=' + index_loc
        utils.create_dir(self.index_dir)
        build_index_command = self.ir_toolkit_location + 'buildindex/IndriBuildIndex'
        toolkit_parameters = [
                                build_index_command,
                                self.parameter_file_location,
                                self.stopwords_file
                                ]

        print(toolkit_parameters)

        proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        (out, err) = proc.communicate()
        print(out.decode("utf-8"))
        print('Index error: ', err)
        if err == None:
            return 'Ok'

class fakeParser:
    def __init__(self):
        self.dataset = 'bioasq' 
#         self.build_index = True
        self.build_index = None
        self.fold = '1'
        self.gen_features = True
#         self.gen_features = None
        

if __name__ == "__main__":
    
    
# #     ir_toolkit_location = sys.argv[1] # '../indri/'

#     # create dataset files dir
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--pool_size', type=int, help='')
    
    args=parser.parse_args()
#     args = fakeParser()
    
    dataset = args.dataset
    workdir = './' + dataset + '_dir/'
    confdir = './' + dataset + '_config/'
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    
#     # generate corpus files to index
    
#     pool_size = 40 # scales very well when server is not used
#     # Get all filenames
    
#     # Options
#     data_dir = '/ssd/francisco/pubmed19-test/'
    if args.dataset == 'robust':
        data_dir = '/ssd/francisco/deep-relevance-ranking/robust04_data/collection/'
    elif args.dataset == 'bioasq':
        data_dir = '/ssd/francisco/pubmed19/'
        
    
    
    
    to_index_dir =  workdir + dataset + '_corpus/'
    index_dir = workdir + dataset + '_indri_index'

    ir_toolkit_location = '../../../indri-l2r/'
    trec_eval_command = '../../eval/trec_eval'
    parameter_file_location = confdir + dataset + '_index_param_file'
    stopwords_file = confdir + 'stopwords'
    
    # Parse Pubmed (BioASQ) dataset
    if args.preprocess:
        print('Preprocessing corpus...')
        if args.dataset == 'bioasq':
            bioasq_corpus_parser.corpus_parser(data_dir, to_index_dir, args.pool_size) # time consuming
        
        ### Robust corpus does not require to apply corpus preprocessing!!

    index_data = Index(ir_toolkit_location, index_dir, parameter_file_location, stopwords_file)
    print('Indexing')
    index_data.build() # time consuming

    