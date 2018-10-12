# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 00:50:31 2018

@author: yishu
"""

import pickle
import collections,re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression


def split_into_lemmas(message): 
    pattern = re.compile('\W')
    string = re.sub(pattern, ' ', str(message)) ## change white space to space
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words] ##lemmantize word

sephora_labeled_sent = pd.read_pickle("sephora_labelled_0929.p")

np.random.seed(29)

## Drop 16 rows with NA 'p_category'
sephora_labeled_sent.dropna(subset=['p_category'], inplace=True)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sephora_labeled_sent[['p_category', 'r_review', 'review_length', 'neg', 'pos', 'r_star']], 
                                                    sephora_labeled_sent['very_helpful'], 
                                                    test_size=0.2, random_state=29)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

text_transformer = Pipeline(steps=[('vect', CountVectorizer(analyzer=split_into_lemmas)),
                                   ('tfidf', TfidfTransformer())])

preprocessor = ColumnTransformer(transformers=[
        ('review', text_transformer, 'r_review'),
        ('numerical', StandardScaler(), ['review_length', 'neg', 'pos', 'r_star']),
        ('cat', OneHotEncoder(), ['p_category'] )])
    
lr_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(random_state=29))])

#lr_parameters = {
#    'preprocessor__review__vect__max_df': [0.5, 0.75],
#    'preprocessor__review__vect__max_features': [5000, 10000],
#    'preprocessor__review__vect__ngram_range': [(1, 2)],
#    'preprocessor__review__tfidf__use_idf': [True],
#    'preprocessor__review__tfidf__norm': ['l2'],
#    'preprocessor__review__tfidf__smooth_idf': [True],
#    'preprocessor__review__tfidf__sublinear_tf': [True],
#    'clf__C': [100, 10, 1, 0.1], 
#    'clf__penalty': ['l2', 'l1'],
#    'clf__tol': [0.0001, 0.001, 0.01, 0.1]
#}


lr_parameters = {
    'preprocessor__review__vect__max_df': [0.5],
    'preprocessor__review__vect__max_features': [10000],
    'preprocessor__review__vect__ngram_range': [(1, 2)],
    'preprocessor__review__tfidf__use_idf': [True],
    'preprocessor__review__tfidf__norm': ['l2'],
    'preprocessor__review__tfidf__smooth_idf': [True],
    'preprocessor__review__tfidf__sublinear_tf': [True],
    'clf__C': [100], 
    'clf__penalty': ['l2'],
    'clf__tol': [ 0.001]
}

lr_grid = GridSearchCV(lr_pipeline, lr_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
lr_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 27690.4882106781
lr_grid.best_score_
# 0.6649691529509069, std: 0.00539128
lr_grid.best_params_
#{'clf__C': 100,
# 'clf__penalty': 'l2',
# 'clf__tol': 0.001,
# 'preprocessor__review__tfidf__norm': 'l2',
# 'preprocessor__review__tfidf__smooth_idf': True,
# 'preprocessor__review__tfidf__sublinear_tf': True,
# 'preprocessor__review__tfidf__use_idf': True,
# 'preprocessor__review__vect__max_df': 0.5,
# 'preprocessor__review__vect__max_features': 10000,
# 'preprocessor__review__vect__ngram_range': (1, 2)}
lr_cv_results=pd.DataFrame(lr_grid.cv_results_)    
# C: 100>10>1>0.1
# max_features: 10000 definitely better.
lr_cv_results.to_pickle("lr_cv_results_withPcategory.p")
