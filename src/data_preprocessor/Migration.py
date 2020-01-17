#!/usr/bin/env python
# coding: utf-8

# # Code to Migrate from this repository!!!   

# ======================
# 
# # Key goals :
# ## 1. Ensure domain names are in sorted order.
# ## 2. Change hscode_6 -> HSCode
# ## 3. Ensure columns of the negative samples follow the same order.
# 
# 
# ======================
# 
# 
# ## Save files in \<Project_Home\>/migration
# 
# 

# In[1]:


import pandas as pd
import os
import sys
import pandas as pd
import numpy as np
import sklearn
import glob
import pickle
import random
from joblib import Parallel, delayed
import yaml
import math
from collections import OrderedDict


# In[ ]:





# In[44]:


def process_part1(DIR):
    SOURCE_DIR = './../../generated_data'
    SOURCE_DIR = os.path.join(SOURCE_DIR, DIR)


    TARGET_DIR = './../../migration/'
    if not os.path.exists(TARGET_DIR): os.mkdir(TARGET_DIR)

    TARGET_DIR = os.path.join(TARGET_DIR,DIR)
    if not os.path.exists(TARGET_DIR): os.mkdir(TARGET_DIR)


    # Read in domain_dims

    with open(os.path.join(SOURCE_DIR, 'domain_dims.pkl'),'rb') as fh:
        dd =  pickle.load(fh)
    # Rename 
    dd['HSCode'] = dd['hscode_6']
    del dd['hscode_6']

    sorted_dd =  OrderedDict({ item[0]:item[1] for item in sorted(dd.items() ,  key=lambda x: x[0]) })
    ordered_domains = list(sorted_dd.keys())

    with open(os.path.join(TARGET_DIR, 'domain_dims.pkl'),'wb') as fh:
        pickle.dump(dd, fh, pickle.HIGHEST_PROTOCOL)
    print('Domains', ordered_domains)    
    # ======================================================================= # 
    # Source files
    # ======================================================================= #
    APE_negative_samples_file = 'negative_samples_ape_1.csv'
    MEAD_negative_samples_file = 'negative_samples_v1.csv'
    test_data_file = 'test_data.csv'
    train_data_file = 'train_data.csv'

    train_df = pd.read_csv(os.path.join(SOURCE_DIR, train_data_file ))
    train_df = train_df.rename(columns={'hscode_6':'HSCode'})
    list_columns = ['PanjivaRecordID']
    list_columns.extend(ordered_domains)
    train_df = train_df[list_columns]
    train_df.to_csv(os.path.join(TARGET_DIR,train_data_file ))

    test_df = pd.read_csv(os.path.join(SOURCE_DIR, test_data_file ))
    test_df = test_df.rename(columns={'hscode_6':'HSCode'})
    list_columns = ['PanjivaRecordID']
    list_columns.extend(ordered_domains)
    test_df = test_df[list_columns]
    test_df.to_csv(os.path.join(TARGET_DIR,test_data_file ))

    mead_neg_samples_df = pd.read_csv( os.path.join(SOURCE_DIR,MEAD_negative_samples_file), low_memory=False )
    mead_neg_samples_df = mead_neg_samples_df.rename(columns={'hscode_6':'HSCode'})
    list_columns = ['PanjivaRecordID', 'NegSampleID']
    list_columns.extend(ordered_domains)
    mead_neg_samples_df = mead_neg_samples_df[list_columns]
    mead_neg_samples_df.to_csv(os.path.join(TARGET_DIR, MEAD_negative_samples_file))


    ape_ns_df= pd.read_csv(os.path.join(SOURCE_DIR,APE_negative_samples_file), low_memory=False)
    ape_ns_df = ape_ns_df.rename(columns={'hscode_6':'HSCode'})
    ape_ns_df.head(10)
    list_columns = ['PanjivaRecordID', 'NegSampleID', 'term_2', 'term_4']
    list_columns.extend(ordered_domains)
    ape_ns_df = ape_ns_df[list_columns]
    ape_ns_df.to_csv(os.path.join(TARGET_DIR, APE_negative_samples_file))
    
    return

# ===================================================================== #

DIR_LIST = ['china_export','us_import2', 'us_import3']
for DIR in DIR_LIST:
    process_part1(DIR)


# In[45]:





# In[ ]:




