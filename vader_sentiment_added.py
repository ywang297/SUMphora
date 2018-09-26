# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 23:39:56 2018

@author: yishu
"""
#import nltk

#nltk.download('vader_lexicon')

import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()

sephora_labeled = pickle.load(open("/Users/yishu/Documents/insight/sephora_labeled.p", "rb"))

sephora_labeled = pd.DataFrame(sephora_labeled)

sephora_nolabel = pickle.load(open("/Users/yishu/Documents/insight/sephora_nolabel.p", "rb"))

sephora_nolabel = pd.DataFrame(sephora_nolabel)


print(sephora_labeled.head())

list(sephora_labeled) #list all variable names in a dataframe

sephora_labeled.info()

sephora_labeled['p_lovecount'].describe()

sephora_labeled.describe().transpose() ## descriptive stats of the variables having numeric values



results_seph = [] ## To keep the dictionaries of scores for all tip text
for idx in tqdm(sephora_labeled.index.values):
    results_seph.append(sia.polarity_scores(sephora_labeled.loc[idx,'r_review'])) 
## The loop above takes 13 seconds on my computer. 
    
## Transform the result to dataframe
results_seph = pd.DataFrame(results_seph) 

## Combine the results with the original data

sephora_labeled_sent = pd.concat([sephora_labeled, results_seph], axis=1)   

sephora_labeled_sent = sephora_labeled_sent[["r_review","p_reviews_recommend", "p_star","helpfulrate","review_length", "bi_helpful", "compound", "neg","neu","pos"]]

sephora_labeled_sent['compound'].describe().transpose()

sephora_labeled_sent.info()

sephora_labeled_sent = sephora_labeled_sent[np.isfinite(sephora_labeled_sent['bi_helpful'])]

sephora_labeled_sent.to_pickle("/Users/yishu/Documents/insight/sephora_labeled_sent.p")




results_seph_nolabel = [] ## To keep the dictionaries of scores for all tip text
for idx in tqdm(sephora_nolabel.index.values):
    results_seph_nolabel.append(sia.polarity_scores(sephora_nolabel.loc[idx, 'r_review'])) 

results_seph_nolabel = pd.DataFrame(results_seph_nolabel) 

sephora_nolabel_sent = pd.concat([sephora_nolabel, results_seph_nolabel], axis=1)   


sephora_nolabel_sent = sephora_nolabel_sent[["r_review","p_reviews_recommend", "p_star","helpfulrate","review_length", "bi_helpful", "compound", "neg","neu","pos"]]

sephora_nolabel_sent.info()

sephora_nolabel_sent = sephora_nolabel_sent[np.isfinite(sephora_nolabel_sent['bi_helpful'])]


sephora_nolabel_sent.to_pickle("/Users/yishu/Documents/insight/sephora_nolabel_sent.p")





