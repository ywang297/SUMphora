#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:59:23 2018

@author: yishu
"""

""" merge all the helpful_reviews with the original file to find the product id"""

import pickle 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

helpful_review = pd.read_pickle("/Users/yishu/Documents/insight/helpful_reviews.p")

sephora= pd.read_pickle("/Users/yishu/Documents/insight/sephora.p")

sephora.info()

sub_sephora = sephora[['p_category', 'p_id', 'p_lovecount', 'p_price'
                       , 'p_star', 'product']]

helpful_review.index.max()


completdat = pd.merge(helpful_review, sub_sephora, left_index=True, right_index=True)

completdat.info()

sephora.loc[sephora.p_id == 'P405402','r_review']

## to check how many reviews in each category
from collections import Counter
my_list = ["red", "blue", "red", "red", "blue"]
print(Counter(my_list))

from gensim.summarization.summarizer import summarize
from tqdm import tqdm
## We can set up a new dataframe called summary with two columns: 'p_id' and 'summary'
p_id_list = list(completdat['p_id'].unique())
summary_list=[]
for p_id in tqdm(p_id_list):
    try:
        reviews_string = ' '.join(list(completdat.loc[completdat['p_id']==p_id, 'r_review'].values))
        summary_list.append(summarize(reviews_string))
    except ValueError:
        summary_list.append(completdat.loc[completdat['p_id']==p_id, 'r_review'].values[0])
    
summary = pd.DataFrame({'p_id': p_id_list, 'summary': summary_list})     

## Or I think maybe it's better to just put a 'summary' column in the completdat dataframe. Code is similar:
for idx in completdat.index.values:
    p_id = completdat.loc[idx, 'p_id']
    reviews_string = ' '.join(list(completdat.loc[completdat['p_id']==p_id, 'r_review'].values))
    completdat.loc[idx, 'summary'] = summarize(reviews_string)


completdat= pd.read_pickle("/Users/yishu/Documents/insight/completdat.p")

test = completdat[completdat['summary']==''] ## 88 lines that do not have summary review 

test[:5]

completdat_new = completdat[~completdat.index.isin(test.index)] ## remove the 88 lines 


