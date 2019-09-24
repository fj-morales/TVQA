# Stronger baseline: Listwise L2R - LambdaMART
# Hyperparameter optimziation HPonsteroids requires Python 3!


# In[2]:


# Imports
import os
import subprocess
import sys

# HPO


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.WARNING)

# HPO server and stuff

# import logging
# logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RS
from hpbandster.examples.commons import MyWorker


# In[2]:


# In[3]:


# Functions
def generate_run_file(pre_run_file, run_file):
    
    with open(pre_run_file, 'rt') as input_f:
        pre_run = input_f.readlines()
#         print('Aqui veo si genero o no el run file: ',type(pre_run))
    with open(run_file, 'wt') as out_f:
        for line in pre_run:
            out_f.write(line.replace('docid=','').replace('indri', 'lambdaMART'))
        


# In[4]:


# Classes
class L2Ranker:
    def __init__(self, ranklib_location, params, normalization=[]):
        self.ranklib_location = ranklib_location
        # Works with Oracle JSE
        # java version "1.8.0_211"
        # Java(TM) SE Runtime Environment (build 1.8.0_211-b12)
        # Java HotSpot(TM) 64-Bit Server VM (build 25.211-b12, mixed mode)
        self.params = params
        self.log_file = ''
        self.ranker_command = ['java', '-jar', ranklib_location + 'RankLib-2.12.jar']
        self.normalization = normalization
        self.save_model_file = ''
        
#     def build(self, ir_tool_params):
    def train(self, train_data_file, save_model_file, config):
        self.save_model_file = save_model_file
        self.log_file = save_model_file + '.log'
        toolkit_parameters = [
                                *self.ranker_command, # * to unpack list elements
                                '-train',
                                train_data_file,
                                *self.normalization,
                                *self.params,
                                '-leaf', 
                                str(config['n_leaves']),
                                '-shrinkage',
                                str(config['learning_rate']),
                                '-tree', # One regression tree per boosted iteration
                                str(config['n_trees']),
                                '-save',
                                self.save_model_file   
                            ] 
        
        print(toolkit_parameters)
        with open(self.log_file, 'wt') as rf:
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
#         proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            
        (out, err)= proc.communicate()
#         print(out.decode('utf-8').splitlines())
#         print(out)
#         print(err)
        print('Model saved: ', self.save_model_file)
            
  

    def gen_run_file(self, test_data_file, run_file):
        pre_run_file = run_file.replace('run_', 'pre_run_', 1)
        toolkit_parameters = [
                                *self.ranker_command, # * to unpack list elements
                                '-load',
                                self.save_model_file,
                                *self.normalization,
                                '-rank',
                                test_data_file,
                                '-indri',
                                pre_run_file     
                            ] 
        
#         print(toolkit_parameters)
        with open(self.log_file, 'at') as rf:
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
#         proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            
        (out, err)= proc.communicate()
#         print(out.decode('utf-8').splitlines())
#         print(out)
#         print(err)

        
        generate_run_file(pre_run_file, run_file)
        
#         print('Run model saved: ', run_file)
