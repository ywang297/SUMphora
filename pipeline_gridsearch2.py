# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 23:34:24 2018

@author: yishu
"""

import pickle
import collections,re
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from wordcloud import WordCloud, STOPWORDS



def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    pattern = re.compile('\W')
    string = re.sub(pattern, '', str(message))
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

sephora_labeled_sent = pd.read_pickle("sephora_labeled_sent_new_new.p")

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sephora_labeled_sent[['review_length', 'compound']].values, 
                                                    sephora_labeled_sent['very_helpful'].values, 
                                                    test_size=0.2, random_state=29)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
lr_pipeline = Pipeline([('clf', LogisticRegression(random_state=29))])
lr_parameters = {
    'clf__C': [1000, 100, 10, 1, 0.1, 0.01, 0.001],
    'clf__penalty': ['l1', 'l2'],
    'clf__tol': [0.0001, 0.001, 0.01, 0.1]
}

lr_grid = GridSearchCV(lr_pipeline, lr_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
lr_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 2.9730477333068848
lr_grid.best_score_
#0.5885574497315257
lr_grid.best_params_
#{'clf__C': 1, 'clf__penalty': 'l1', 'clf__tol': 0.0001}

""" Try with 'neg', 'neu', 'pos' instead of 'compound' """

x_train, x_test, y_train, y_test = train_test_split(sephora_labeled_sent[['review_length', 'neg', 'neu', 'pos']].values, 
                                                    sephora_labeled_sent['very_helpful'].values, 
                                                    test_size=0.2, random_state=29)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
lr_pipeline = Pipeline([('clf', LogisticRegression(random_state=29))])
lr_parameters = {
    'clf__C': [1000, 100, 10, 1, 0.1, 0.01, 0.001],
    'clf__penalty': ['l1', 'l2'],
    'clf__tol': [0.0001, 0.001, 0.01, 0.1]
}

lr_grid = GridSearchCV(lr_pipeline, lr_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
lr_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 3.516585350036621
lr_grid.best_score_
#0.5699961365235611
lr_grid.best_params_
#{'clf__C': 0.1, 'clf__penalty': 'l1', 'clf__tol': 0.0001}

""" colinearity? do not put 'neu' """
x_train, x_test, y_train, y_test = train_test_split(sephora_labeled_sent[['review_length', 'neg', 'pos']].values, 
                                                    sephora_labeled_sent['very_helpful'].values, 
                                                    test_size=0.2, random_state=29)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
lr_pipeline = Pipeline([('clf', LogisticRegression(random_state=29))])
lr_parameters = {
    'clf__C': [1000, 100, 10, 1, 0.1, 0.01, 0.001],
    'clf__penalty': ['l1', 'l2'],
    'clf__tol': [0.0001, 0.001, 0.01, 0.1]
}

lr_grid = GridSearchCV(lr_pipeline, lr_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
lr_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time:  3.154561996459961
lr_grid.best_score_
#0.5700145635884414
lr_grid.best_params_
#{'clf__C': 0.1, 'clf__penalty': 'l1', 'clf__tol': 0.0001}

""" Add in 'r_star' to see.. """
x_train, x_test, y_train, y_test = train_test_split(sephora_labeled_sent[['review_length', 'neg', 'pos', 'r_star']].values, 
                                                    sephora_labeled_sent['very_helpful'].values, 
                                                    test_size=0.2, random_state=29)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
lr_pipeline = Pipeline([('clf', LogisticRegression(random_state=29))])
lr_parameters = {
    'clf__C': [1000, 100, 10, 1, 0.1, 0.01, 0.001],
    'clf__penalty': ['l1', 'l2'],
    'clf__tol': [0.0001, 0.001, 0.01, 0.1]
}

lr_grid = GridSearchCV(lr_pipeline, lr_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
lr_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time:  5.032538890838623
lr_grid.best_score_
#0.62020710218147
lr_grid.best_params_
#{'clf__C': 10, 'clf__penalty': 'l2', 'clf__tol': 0.01}

""" What if we look at 'accurcy' when grid search? """
lr_grid = GridSearchCV(lr_pipeline, lr_parameters, cv=5, n_jobs=1, scoring='accuracy', verbose=1)

import time
t0=time.time()
lr_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time:  4.5358664989471436
lr_grid.best_score_
#0.6319263666790629 
#For comparison, the expectation of accuracy of all 1 prediction should be 7235/13445=0.53811826
lr_grid.best_params_
#{'clf__C': 100, 'clf__penalty': 'l2', 'clf__tol': 0.001}