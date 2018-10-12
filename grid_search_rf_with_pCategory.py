# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 00:40:30 2018

@author: yishu
"""

import pickle
import collections,re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    pattern = re.compile('\W')
    string = re.sub(pattern, '', str(message))
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

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

text_transformer = Pipeline(steps=[('vect', CountVectorizer()),
                                   ('tfidf', TfidfTransformer())])

preprocessor = ColumnTransformer(transformers=[
        ('review', text_transformer, 'r_review'),
        ('numerical', StandardScaler(), ['review_length', 'neg', 'pos', 'r_star']),
        ('cat', OneHotEncoder(), ['p_category'] )])
    
    
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=29))]) 

#rf_parameters = {
#    'preprocessor__review__vect__max_df': [0.5, 0.75],
#    'preprocessor__review__vect__max_features': [5000, 10000, None],
#    'preprocessor__review__vect__ngram_range': [(1, 2)],  # unigrams or bigrams
#    'preprocessor__review__tfidf__use_idf': [True],
#    'preprocessor__review__tfidf__norm': ['l2'],
#    'preprocessor__review__tfidf__smooth_idf': [True],
#    'preprocessor__review__tfidf__sublinear_tf': [True],
#    'clf__n_estimators': [20, 50, 100],
#    'clf__max_depth': [4, 5],
#    #'clf__min_samples_split': [5, 10, 20],
#    'clf__min_samples_leaf': [1, 2, 5], 
#    'clf__bootstrap': [True]
#}   


rf_parameters = {
    'preprocessor__review__vect__max_df': [0.5],
    'preprocessor__review__vect__max_features': [5000],
    'preprocessor__review__vect__ngram_range': [(1, 2)],  # unigrams or bigrams
    'preprocessor__review__tfidf__use_idf': [True],
    'preprocessor__review__tfidf__norm': ['l2'],
    'preprocessor__review__tfidf__smooth_idf': [True],
    'preprocessor__review__tfidf__sublinear_tf': [True],
    'clf__n_estimators': [100],
    'clf__max_depth': [4],
    #'clf__min_samples_split': [5, 10, 20],
    'clf__min_samples_leaf': [1], 
    'clf__bootstrap': [True]
} 

rf_grid = GridSearchCV(rf_pipeline, rf_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
rf_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 23634.023411512375
rf_grid.best_score_
# 0.6461864877358034, std: 0.00572829
rf_grid.best_params_
#{'clf__bootstrap': True,
# 'clf__max_depth': 4,
# 'clf__min_samples_leaf': 1,
# 'clf__n_estimators': 100,
# 'preprocessor__review__tfidf__norm': 'l2',
# 'preprocessor__review__tfidf__smooth_idf': True,
# 'preprocessor__review__tfidf__sublinear_tf': True,
# 'preprocessor__review__tfidf__use_idf': True,
# 'preprocessor__review__vect__max_df': 0.5,
# 'preprocessor__review__vect__max_features': 5000,
# 'preprocessor__review__vect__ngram_range': (1, 2)}
rf_cv_results=pd.DataFrame(rf_grid.cv_results_)    
# n_estimators: 100 definitely better
# min_samples_leaf: 1 definitely better
rf_cv_results.to_pickle("rf_cv_results_withPcategory.p")


probs_rf = rf_grid.predict_proba(x_test)
preds_rf = probs_rf[:,1]
fpr_rf, tpr_rf, threshold = metrics.roc_curve(y_test, preds_rf)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
