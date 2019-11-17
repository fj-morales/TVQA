# Imports
import os
import subprocess
import sys

# REMOVE!!
from ir_utils import *

# model

from ir_lmart import *

# HPO

# from hpo import *


# from HpoWorker import *
from HpoWorker import *

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# from hpbandster.core.worker import Worker

# HPO server and stuff

# import logging
# logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RS

import random
import pickle

from ir_test import *

from datetime import datetime

import logging
logging.basicConfig(level=logging.WARNING)
# In[2]:


# In[3]:


# In[2]:


def tickets(max):
    return {i:i for i in range(1,max)}


# In[3]:




# In[5]:


def get_train_budget_data_file(budget, query_list, train_data_file):
    # Budget is percentage of training data: 
    # min_budget = 10%
    # max_budget = 100%
    if (int(budget) <= 100 or int(budget) >= 10):
        len_queries = len(query_list)
        budgeted_queries = round(len_queries * (budget / 100))
        print('total budget:', len_queries)
        print('allocated budget:', budgeted_queries)
        train_budget_queries_file = train_data_file + '_budget' + str(budget)
        if not os.path.exists(train_budget_queries_file):
            with open(train_data_file, 'rt') as f_in:
                with open(train_budget_queries_file, 'wt') as budget_file_out:
                    for query_feature in f_in:                        
                        qid = query_feature.split()[1].split(':')[1]
#                         print(qid)
                        if qid in query_list[0:budgeted_queries]:
                            
                            budget_file_out.write(query_feature)
        else:
            print("File already exists")
            return train_budget_queries_file                
    else:
        print('Budget is outside the limits (10% < b < 100%): ', budget)
        return 


# In[6]:


# Functions
def generate_run_file(pre_run_file, run_file):
    
    with open(pre_run_file, 'rt') as input_f:
        pre_run = input_f.readlines()
#         print(type(pre_run))
    with open(run_file, 'wt') as out_f:
        for line in pre_run:
            out_f.write(line.replace('docid=','').replace('indri', 'lambdaMART'))
        
# In[4]:


# In[7]:


class fakeParser:
    def __init__(self):
        self.dataset = 'bioasq' 
        self.min_budget = 10
        self.max_budget = 10
        self.n_iterations = 1
        self.n_workers = 1
        self.hpo_method = 'rs'


# In[8]:


# Main
if __name__ == "__main__":
    
    # Options and variables

    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--hpo_method',   type=str, help='')
    parser.add_argument('--min_budget',   type=int, help='Minimum (percentage) budget used during the optimization.',    default=10)
    parser.add_argument('--max_budget',   type=int, help='Maximum (percentage) budget used during the optimization.',    default=100)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=500)
    parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=5)
    parser.add_argument('--default_config', action='store_true', )
    parser.add_argument('--norm', action='store_true' )
    parser.add_argument('--test_mode', action='store_true', help='Test specific hyperparameter configuration' )
    parser.add_argument('--leaf', type=int, help='Number of leaves. Only works with test mode.' , default=10)
    parser.add_argument('--lr', type=float, help='Learning rate. Only works with test mode.' , default=0.1)
    parser.add_argument('--tree', type=int, help='Number of trees. Only works with test mode.' , default=1000)
    parser.add_argument('--previous_run_dir',type=str, help='A directory that contains a config.json and results.json for the same configuration space.', default=None)
    
    args=parser.parse_args()
#     args = fakeParser()
    dataset = 'tvqa'
    workdir = './workdir/'
    
    
    # current date and time
    now = datetime.now()
    timestamp = int(datetime.timestamp(now))
    if args.test_mode:
        args.default_config = True
        
    
    if args.default_config:
        print('Using default HPO config values and only one worker, iteration, max_budget, and rs method.\n')
        args.hpo_method = 'rs'
        args.min_budget = 100
        args.max_budget = 100
        args.n_iterations = 1
        args.n_workers = 1    
    

    hpo_method = args.hpo_method
    hpo_results_dir = workdir + 'hpo_results' + '_' + hpo_method + '_' + str(timestamp) + '/'    
    
    inter_results_file = dataset + '_results_' + hpo_method + '_' + str(timestamp) + '.pkl'
    
    if (args.default_config == False) and (args.test_mode == False):
        result_logger = hpres.json_result_logger(directory=hpo_results_dir, overwrite=False)
    else:
        result_logger = None
    
        
    million_tickets = tickets(1000000)

    confdir = './tvqa_config/'
    gen_features_dir = workdir + 'gen_features_dir/'
    ranklib_location = '../ranklib/'
    
#     trec_eval_command = '../../eval/trec_eval'
    
    metric2t = 'P@1' # 'MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k (default=ERR@10)'
    
    ranker_type = '6' # LambdaMART
    
    # normalization: Feature Engineering?
    if args.norm:
        norm_params = ['-norm', 'zscore'] # 'sum', 'zscore', 'linear'
    else:
        norm_params = [] 
    
    if hpo_method == 'rs':
        hpo_run_id = "RandomSearch"
        # Start a nameserver (see example_3)
        NS = hpns.NameServer(run_id = hpo_run_id, host='127.0.0.1', port=0)
        ns_host, ns_port = NS.start()
    elif hpo_method == 'bohb':
        hpo_run_id = "BOHB"
        NS = hpns.NameServer(run_id = hpo_run_id, host='127.0.0.1', port=0)
        ns_host, ns_port = NS.start()

    print(ns_host,':', ns_port)
        
    workers=[]
    for i in range(args.n_workers):
        worker = HpoWorker(dataset, workdir, confdir, gen_features_dir, ranklib_location, norm_params, ranker_type,
                           metric2t, million_tickets, nameserver=ns_host, nameserver_port=ns_port, run_id=hpo_run_id, id=i)
        worker.run(background=True)
        workers.append(worker)

    # Continue previous runs
    if args.previous_run_dir:
        previous_run = hpres.logged_results_to_HBS_result(args.previous_run_dir)
    else:
        previous_run = None
        
    # Random search

    if hpo_method == 'rs':
        hpo_worker = RS(  configspace = worker.get_configspace(args.default_config, 
                                                       args.test_mode,
                                                       args.leaf,
                                                       args.lr,
                                                       args.tree
                                                      ),
                              run_id = hpo_run_id, 
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              result_logger=result_logger,
                              min_budget = args.max_budget, max_budget = args.max_budget,
                              previous_result = previous_run
                       )
        res = hpo_worker.run(n_iterations = args.n_iterations, min_n_workers = args.n_workers)

    elif hpo_method == 'bohb':
        hpo_worker = BOHB(  configspace = worker.get_configspace(args.default_config, 
                                                           args.test_mode,
                                                           args.leaf,
                                                           args.lr,
                                                           args.tree
                                                          ),

                              run_id = hpo_run_id, 
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              result_logger=result_logger,
                              min_budget = args.min_budget, max_budget = args.max_budget,
                              previous_result = previous_run
                       )
        res = hpo_worker.run(n_iterations = args.n_iterations, min_n_workers = args.n_workers)

    if (args.default_config == False) and (args.test_mode == False):
        # store results
        with open(os.path.join(hpo_results_dir, inter_results_file), 'wb') as fh:
            pickle.dump(res, fh)
            
    hpo_worker.shutdown(shutdown_workers=True)


    # In[14]:


    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.


    # In[12]:


    NS.shutdown()


    # In[13]:

    # Evaluate
    
    test_results = test_model(workdir, ranklib_location, norm_params, res)

    print('BEST RESULTS EVER!!: ', test_results)
    

    # Save results for further_analysis

    if args.default_config:
        results_file = workdir + dataset + '_' + 'defaults' + '_' + str(timestamp) +'.pickle'
    else:
        results_file = workdir + dataset + '_' + 'hpo_results_' + hpo_method + '_' + str(timestamp) +'.pickle'

    results = {'hpo_config': args,
                'hpo_results': res

    }

    pickle.dump(results, open(results_file, "wb" ) )


    # In[14]:


    # In[15]:


    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()



    # In[16]:


    all_runs = res.get_all_runs()



    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
    print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))


    # In[18]:


    #     qrels_test_file = './bioasq_dir/bioasq.test_qrels'
    #     run_val_file = './this.file'
    #     lmart_model = res['info']
    #     lmart_model.gen_run_file(test_data_file, run_val_file)
    #     eval(trec_eval_command, qrels_test_file, run_val_file)


    # In[10]:


    # In[15]:


    res.get_incumbent_id()


    # In[16]:


    id2config[res.get_incumbent_id()]


    # In[17]:


    # res.get_learning_curves()


    # In[18]:


    res.get_all_runs()


    # In[19]:


    res.get_id2config_mapping()

