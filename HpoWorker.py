import random
import itertools
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

from ir_utils import *
from ir_preprocessing import load_json

# HPO server and stuff

# import logging
# logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres


from hpbandster.examples.commons import MyWorker
import copy

from eval_utils import *
import numpy as np

import numpy as np

# model

from ir_lmart import *

import logging
logging.basicConfig(level=logging.WARNING)

# def get_train_budget_data_file(budget, qid_list, train_data_file):
#     # Budget is percentage of training data: 
#     # min_budget = 10%
#     # max_budget = 100%
#     if (int(budget) <= len(qid_list)):
#         len_queries = len(qid_list)
    
# #         print('total budget:', len_queries)
# #         print('allocated budget:', budget)
#         train_budget_queries_file = train_data_file + '_budget' + str(budget)
#         if not os.path.exists(train_budget_queries_file):
#             #                         print('queries lenght\n:',len_queries)
#             with open(train_data_file, 'rt') as f_in:
#                 with open(train_budget_queries_file, 'wt') as budget_file_out:
                    
#                     for query_feature in f_in:                        

#                         qid = query_feature.split()[1].split(':')[1]
# #                         print(qid)
#                         if qid in qid_list[0:budget]:
                            
#                             budget_file_out.write(query_feature)
#                     return train_budget_queries_file
#         else:
# #             print("File already exists")
#             return train_budget_queries_file                
#     else:
#         print('Budget is outside the limits (10% < b < 100%): ', budget)
#         return 'THIS IS NOT WORKING'

def get_train_budget_data_file(budget, train_questions_file, train_data_file):
    # Budget is percentage of training data: 
    # min_budget = 10%
    # max_budget = 100%
    
    train_budget_queries_file = train_data_file + '_budget' + str(budget)
    if not os.path.exists(train_budget_queries_file):

        if (int(budget) <= 100 or int(budget) >= 10):
            # Preparing budgeted train features data 
            train_questions = load_json(train_questions_file)
            qid_list = [str(x['qid']) for x in train_questions]
            len_queries = len(qid_list) 
            
            budgeted_queries = round(len_queries * (budget / 100))
            print('total budget:', len_queries)
            print('allocated budget:', budgeted_queries)
            
            with open(train_data_file, 'rt') as f_in:
                with open(train_budget_queries_file, 'wt') as budget_file_out:
                    for query_feature in f_in:                        
                        qid = query_feature.split()[1].split(':')[1]
                        
                        if qid in qid_list[0:budgeted_queries]:
                            
                            budget_file_out.write(query_feature)
                    return train_budget_queries_file

        else:
            print('Budget is outside the limits (10% < b < 100%): ', budget)
            return 'THIS IS NOT WORKING'   
    else:
        print("File already exists")
        return train_budget_queries_file                
    
def compute_one_fold(budget, config, tickets, save_model_prefix, run_file_prefix, model_instance, 
                     train_questions_file, train_data_file, gold_answer_qrels_file, *args, **kwargs):
    
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """
            
            budget = int(budget)
            
            my_ticket = None
            
            while my_ticket is None:

                try:
                    this_key = list(tickets.keys())[0]
                    my_ticket = tickets.pop(this_key)
                    print('Run ID:' , my_ticket, '\n')
                except:
                    pass
            
            
#             print('My ticket: ', my_ticket, '\n')
#             print('Tickets left: ', len(self.tickets), '\n')
            
            
            
            # Train model with config parameters
                
            #     pre_run_file = workdir + 'pre_run_' + dataset + l2r_model
            
            config['learning_rate'] = round(config['learning_rate'],2)
            n_l = config['n_leaves']
            l_r = config['learning_rate']
            n_t = config['n_trees']
            
            config_suffix = 'id' + str(my_ticket) + '_budget' + str(budget) + '_leaves' + str(n_l) + '_lr' + str(l_r) + '_n' + str(n_t)
            
            save_model_file = save_model_prefix + config_suffix
            
            run_val_file = run_file_prefix + config_suffix
            
#             print('Type class of budget variable: ', type(budget))
#             print(self.qid_list)
            

            budget_train_features_file = get_train_budget_data_file(budget, train_questions_file, train_data_file)
            
#             lmart_model = L2Ranker(ranklib_location, l2r_params, norm_params)
#             print('budget_file:', budget_train_features_file)
        
            model = copy.deepcopy(model_instance)
            model.train(budget_train_features_file, save_model_file, config)
        
            val_data_file = model.params[1]
            
            model.gen_run_file(val_data_file, run_val_file)
            
            # Evaluate Model
            
#             [pred_answers, gold_answers] = load_predictions(run_val_file, val_ids_equiv_file, val_questions_file) # before
            [pred_answers, gold_answers] = load_predictions(run_val_file, gold_answer_qrels_file)
            
            val_acc = evaluate(pred_answers, gold_answers)

            
            print('Metric: ', val_acc, '\n')
            
            #import IPython; IPython.embed()
            return ({
                    'metric': val_acc,
                    'info': { 'label': 'Accuracy based, LambdaMART optimized based on P@1',
                             'model_file':  save_model_file
                        }
            })


class HpoWorker(Worker):
    def __init__(self, dataset, workdir, confdir, gen_features_dir, ranklib_location, norm_params, ranker_type, metric2t, tickets, **kwargs):
            super().__init__(**kwargs)
            self.dataset = dataset
            self.workdir = workdir
            self.ranklib_location = ranklib_location
            self.tickets = tickets  
            self.ranker_type = ranker_type
            self.metric2t = metric2t
            self.norm_params = norm_params
            self.confdir = confdir
            self.gen_features_dir = gen_features_dir
            
#             self.save_model_prefix = save_model_prefix
#             self.run_file_prefix = run_file_prefix
#             self.model = copy.deepcopy(model_instance)
#             self.budget_train_features_file = budget_train_features_file
#             self.qid_list = qid_list
#             self.trec_eval_command = trec_eval_command
#             self.qrels_val_file = qrels_val_file
            
    def compute(self, config, budget, *args, **kwargs):
    
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """

#             if self.dataset == 'bioasq':
#                 folds = ['']
#             elif self.dataset == 'robust':
#                 folds = ['1','2','3','4','5']


            folds = ['']
            cv_results_dict = {}
            fold_dir = self.workdir
            for fold in folds:
                
                dataset_fold = self.dataset
                train_data_file = self.gen_features_dir + 'l2r_features_train'
                val_data_file = self.gen_features_dir + 'l2r_features_dev'
                test_data_file = self.gen_features_dir + 'l2r_features_test'
#                 qrels_val_file = self.workdir + 'gold_answer_qrels_dev'
#                 all_data_ids_equiv_file = self.workdir + 'all_data_ids_equiv.json'
                train_questions_file = './data/tvqa_new_train_processed.json'
                val_questions_file = './data/tvqa_new_dev_processed.json'
                val_ids_equiv_file = self.workdir + 'dev_ids_equiv.json'
                gold_answer_qrels_file = self.workdir + 'gold_answer_qrels_' + 'dev'
                            
                if self.ranker_type == '6':
                    l2r_model = '_lmart_'
                    
                enabled_features_file = self.confdir + self.dataset + l2r_model + 'enabled_features'
                l2r_params = [
                    '-validate',
                    val_data_file,
                    '-ranker',
                    self.ranker_type,
                    '-metric2t',
                    self.metric2t,
                    '-feature',
                    enabled_features_file
                ]

                # Run train
                lmart_model = L2Ranker(self.ranklib_location, l2r_params, self.norm_params)

                save_model_prefix = fold_dir + dataset_fold + l2r_model

                run_file_prefix = fold_dir + 'retrieved_files/' + 'run_' + dataset_fold + l2r_model

                train_features_file = fold_dir + self.dataset + '_' + 'train' + '_features'


#                 budget_train_features_file = train_data_file


                # Compute results for one fold
                one_fold_results = compute_one_fold(budget, config, self.tickets, save_model_prefix, run_file_prefix, lmart_model, 
                                                     train_questions_file, train_data_file, gold_answer_qrels_file)

                cv_results_dict['s' + fold] = one_fold_results

            cv_mean_metric = round(np.mean([value['metric'] for key,value in cv_results_dict.items()]), 8)
            cv_std_metric = round(np.std([value['metric'] for key,value in cv_results_dict.items()]), 8)
            
            cv_results_dict['mean_metric'] = cv_mean_metric
            cv_results_dict['std_metric'] = cv_std_metric
            
            
            return ({
                'loss': 1 - cv_mean_metric, # remember: HpBandSter always minimizes!
                'info': cv_results_dict
            })                
            
        
    @staticmethod
    def get_configspace(default_config,test_mode,leaf,lr,tree):
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()
            
            n_leaves = CSH.UniformIntegerHyperparameter('n_leaves', lower=5, upper=100, default_value=10, q=5, log=False)
            learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=0.5, default_value=0.1, q=0.01, log=False)
            n_trees = CSH.UniformIntegerHyperparameter('n_trees', lower=100, upper=2000, default_value=1000, q=50 ,log=False)
            
            
#             n_leaves = CSH.UniformIntegerHyperparameter('n_leaves', lower=10, upper=11, default_value=10, log=False)
#             learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.1, upper=0.2, default_value=0.1, q=0.1, log=False)
#             n_trees = CSH.UniformIntegerHyperparameter('n_trees', lower=1, upper=2, default_value=1, q=1, log=False)

            if default_config:
                n_leaves = CSH.OrdinalHyperparameter('n_leaves', sequence=[10])
                learning_rate = CSH.OrdinalHyperparameter('learning_rate', sequence=[0.1])
                n_trees = CSH.OrdinalHyperparameter('n_trees', sequence=[1000])
            
            if test_mode:
                n_leaves = CSH.OrdinalHyperparameter('n_leaves', sequence=[leaf])
                learning_rate = CSH.OrdinalHyperparameter('learning_rate', sequence=[lr])
                n_trees = CSH.OrdinalHyperparameter('n_trees', sequence=[tree])            
            
            cs.add_hyperparameters([n_leaves, learning_rate, n_trees])

            return cs

if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)