# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 23:39:56 2018

@author: yishu
"""
#import nltk

#nltk.download('vader_lexicon')

import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()

sephora = pickle.load(open("/Users/yishu/Documents/insight/sephora_retry.p", "rb"))

sephora = pd.DataFrame(sephora)

print(sephora.head())

list(sephora) #list all variable names in a dataframe

sephora.info()

sephora['p_lovecount'].describe()

sephora.describe().transpose() ## descriptive stats of the variables having numeric values



results_seph = [] ## To keep the dictionaries of scores for all tip text
for i in tqdm(np.arange(len(sephora))):
    results_seph.append(sia.polarity_scores(sephora.loc[i, 'r_review'])) 
## The loop above takes 13 seconds on my computer. 
    
## Transform the result to dataframe
results_seph = pd.DataFrame(results_seph) 

## Combine the results with the original data

sephora_sent = pd.concat([sephora, results_seph], axis=1)   
sephora_sent.to_pickle("/Users/yishu/Documents/insight/sephora_sentiment.p")

sephora_sent['compound'].describe().transpose()


sephora_sent.info()







