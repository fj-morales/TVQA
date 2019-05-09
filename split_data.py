#!/usr/bin/env python
# coding: utf-8

# ## Split data 

# In[4]:


import json


# In[5]:


if __name__ == '__main__':
    train_file = './data/tvqa_train_processed.json'

    with open(train_file, 'r') as f:
        processed_data_train = json.load(f)
    
    train_split = int(0.8 * len(processed_data_train))
    
    new_train = processed_data_train[0:train_split]
    new_dev = processed_data_train[train_split:]
    with open('./data/tvqa_new_train_processed.json', 'wt') as new_train_f:
        json.dump(new_train, new_train_f, indent=4)
    
    with open('./data/tvqa_new_dev_processed.json', 'wt') as new_dev_f:
        json.dump(new_dev, new_dev_f, indent=4)


# In[6]:


with open('./data/tvqa_new_train_processed.json') as f:
    data_open = json.load(f)

