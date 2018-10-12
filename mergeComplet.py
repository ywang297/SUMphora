#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 07:14:36 2018

@author: yishu
"""

""" merge all the helpful_reviews with the original file to find the product id"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

helpful_review = pd.read_pickle("helpful_reviews.p")

sephora = pd.read_pickle("sephora.p")

sephora.info()

sub_sephora = sephora[['p_category', 'p_id', 'p_lovecount', 'p_price', 'p_star', 'product']]

helpful_review.index.max()


completdat = pd.merge(helpful_review, sub_sephora, left_index=True, right_index=True)

completdat.info()


""" Redo the summarization to incorporate the length of total reviews for different products """
from gensim.summarization.summarizer import summarize
from tqdm import tqdm
import re
for idx in tqdm(completdat.index.values):
    p_id = completdat.loc[idx, 'p_id']
    reviews = list(completdat.loc[completdat['p_id']==p_id, 'r_review'].values)
    n_reviews = len(reviews)
    reviews_string = ' '.join(reviews)
    n_sentences = len(re.split("[.?!] ", reviews_string))
    #print('number of reviews:', n_reviews, 'number of sentences:', n_sentences)
    if n_sentences >= 200:
        n_keep_sen = 20
        alpha = n_keep_sen/n_sentences
    elif n_sentences >=80:
        alpha = 0.1
    elif n_sentences >=60:
        alpha = 0.12
    elif n_sentences >=40:
        alpha = 0.15
    elif n_sentences >=20:
        alpha = 0.2
    elif n_sentences >=10:
        alpha = 0.25
    else:
        alpha = 0.3
    try:
        completdat.loc[idx, 'summary'] = summarize(reviews_string, ratio=alpha)
    except ValueError:
        completdat.loc[idx, 'summary'] = ''
        
completdat[completdat['summary']==''].shape[0]
## 65
## Now ONLY 65 rows (corresponding to 48 different p_ids) having summary == '', why???? 
## Maybe it's correct, since now for p_id with few sentences, e.g. those have <10 sentences,  
## since alpha =0.3, it only needs 3-4 sentences to return 1 sentence, so fewer '' rows.
## Is this the correct reasoning???

completdat.to_pickle("completdat_new_1003.p")



completdat_new = pd.read_pickle("/Users/yishu/Documents/insight/completdat_new_1003.p")

completdat_new.summary[0]


test= completdat_new.loc[completdat_new['product']=="Calendula Deep Clean Foaming Face Wash", 'r_review'].tolist()

test0= completdat_new.loc[completdat_new['product']=="Calendula Deep Clean Foaming Face Wash", 'summary'].tolist()[0]


lentest = len(test)

test_string = ' '.join(test)
test_sentences = len(re.split("[.?!] ", test_string))

test_summary = ''.join(test0)
summary_sentences = len(re.split("[.?!] ", test_summary))


completdat_new.loc[completdat_new['product']=="Calendula Deep Clean Foaming Face Wash", 'p_id'].count()


''' try to get the plot '''

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
#import urllib.request
#from urllib.request import urlopen
import requests


print(test0)









