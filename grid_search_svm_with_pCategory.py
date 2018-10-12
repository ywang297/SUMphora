# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:48:32 2018

@author: yishu
"""

import pickle
import collections,re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    pattern = re.compile('\W')
    string = re.sub(pattern, ' ', str(message))
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

sephora_labeled_sent = pd.read_pickle("/Users/yishu/Documents/insight/sephora_labelled_0929.p")

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

text_transformer = Pipeline(steps=[('vect', CountVectorizer()),
                                   ('tfidf', TfidfTransformer())])

preprocessor = ColumnTransformer(transformers=[
        ('review', text_transformer, 'r_review'),
        ('numerical', StandardScaler(), ['review_length', 'neg', 'pos', 'r_star']),
        ('cat', OneHotEncoder(), ['p_category'] )])
    
from sklearn.svm import SVC    
    
svm_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', SVC(probability=True, random_state=29))]) 

#svm_parameters = {
#    'preprocessor__review__vect__max_df': [0.5, 0.75],
#    'preprocessor__review__vect__max_features': [5000, 10000, None],
#    'preprocessor__review__vect__ngram_range': [(1, 2)],  # unigrams or bigrams
#    'preprocessor__review__tfidf__use_idf': [True],
#    'preprocessor__review__tfidf__norm': ['l2'],
#    'preprocessor__review__tfidf__smooth_idf': [True],
#    'preprocessor__review__tfidf__sublinear_tf': [True],
#    'clf__C': [5, 1, 0.2],
#    'clf__gamma': [0.05, 0.10, 0.2],
#    'clf__shrinking': [True],
#    'clf__tol': [0.001, 0.0001], 
#    'clf__class_weight': [None]
#}   



svm_parameters = {
    'preprocessor__review__vect__max_df': [0.5],
    'preprocessor__review__vect__max_features': [10000],
    'preprocessor__review__vect__ngram_range': [(1, 2)],  # unigrams or bigrams, change 2 to 3 does not help the fit
    'preprocessor__review__tfidf__use_idf': [True],
    'preprocessor__review__tfidf__norm': ['l2'],
    'preprocessor__review__tfidf__smooth_idf': [True],
    'preprocessor__review__tfidf__sublinear_tf': [True],
    'clf__C': [5],
    'clf__gamma': [0.10],
    'clf__shrinking': [True],
    'clf__tol': [0.001], 
    'clf__class_weight': [None]
} 



svm_grid = GridSearchCV(svm_pipeline, svm_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
svm_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 26891.162678956985
svm_grid.best_score_
# 0.6762689774909967 std 0.00753464
svm_grid.best_params_
#{'clf__C': 5,
# 'clf__class_weight': None,
# 'clf__gamma': 0.1,
# 'clf__shrinking': True,
# 'clf__tol': 0.001,
# 'preprocessor__review__tfidf__norm': 'l2',
# 'preprocessor__review__tfidf__smooth_idf': True,
# 'preprocessor__review__tfidf__sublinear_tf': True,
# 'preprocessor__review__tfidf__use_idf': True,
# 'preprocessor__review__vect__max_df': 0.5,
# 'preprocessor__review__vect__max_features': 10000,
# 'preprocessor__review__vect__ngram_range': (1, 2)}

svm_cv_results=pd.DataFrame(svm_grid.cv_results_)
# C: 5 better than 1 better than 0.2
# tol: 0.001 slightly better than 0.0001
# gamma: 0.1 better than 0.2 and 0.05
# max_features: 10000 and None same, both better than 5000
# max_df: 0.5 and 0.75 no difference
svm_cv_results.to_pickle("svm_cv_results_withPcategory.p")
