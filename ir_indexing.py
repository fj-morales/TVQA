# IR_indexing

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import subprocess
import sys

import argparse

from ir_utils import *

## My libraries


# In[ ]:


class Index:
    def __init__(self, ir_toolkit_location, index_dir, parameter_file_location):
        self.ir_toolkit_location = ir_toolkit_location
        self.parameter_file_location = parameter_file_location
        self.index_dir = index_dir
#         self.stopwords_file = stopwords_file
#     def build(self, ir_tool_params):
    def build(self):
        
    #     index_loc_param = '--indexPath=' + index_loc
        create_dir(self.index_dir)
        build_index_command = self.ir_toolkit_location + 'buildindex/IndriBuildIndex'
        toolkit_parameters = [
                                build_index_command,
                                self.parameter_file_location
#                                 self.stopwords_file
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
    
    

#     # create dataset files dir
    
#     parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
#     parser.add_argument('--preprocess', action='store_true')
#     parser.add_argument('--dataset',   type=str, help='')
#     parser.add_argument('--pool_size', type=int, help='')
    
#     args=parser.parse_args()
#     args = fakeParser()
    
    
    ir_toolkit_location = '../indri-l2r/'
    workdir = './workdir/'
    confdir = './tvqa_config/'

    to_index_input = workdir + 'index_inputs/'
    index_dir = workdir + 'index_dir/'

    parameter_file_location = confdir + 'tvqa_index_param_file'

    index_data = Index(ir_toolkit_location, index_dir, parameter_file_location)
    print('Indexing')
    index_data.build() # time consuming

    