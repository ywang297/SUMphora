# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:59:09 2018

@author: yishu
"""

""" Use one of the best models to predict on test set, combine those predicted to be 1 (helpful) with 
those with 1 in training set to make the full helpful review set for the next step. """

import pickle
import collections,re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

np.random.seed(29)

sephora_labeled_sent = pd.read_pickle("sephora_labeled_sent_new_new.p")
sephora_nolabel_sent = pd.read_pickle("sephora_nolabel_sent_new_new.p")

xs = sephora_labeled_sent[['r_review', 'review_length', 'neg', 'pos', 'r_star']]
ys = sephora_labeled_sent['very_helpful']

x_test = sephora_nolabel_sent[['r_review', 'review_length', 'neg', 'pos', 'r_star']]


def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    pattern = re.compile('\W')
    string = re.sub(pattern, '', str(message))
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]


text_transformer_best = Pipeline(steps=[('vect', CountVectorizer(analyzer=split_into_lemmas, max_df=0.5, max_features=5000, ngram_range=(1,2))),
                                   ('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=True, use_idf=True))])


preprocessor_best = ColumnTransformer(transformers=[
        ('review', text_transformer_best, 'r_review'),
        ('numerical', StandardScaler(), ['review_length', 'neg', 'pos', 'r_star'])])
    
svm_pipeline_best = Pipeline(steps=[('preprocessor', preprocessor_best), ('clf', SVC(random_state=29, C=1, gamma=0.1,
                                                          tol=0.001, shrinking=True, class_weight=None))])   
    
svm_pipeline_best.fit(xs, ys)
y_pred = svm_pipeline_best.predict(x_test) 
#len(y_pred)
#10122
#sum(y_pred)
#8303
#sum(y_pred)/len(y_pred)
#0.8202924323256273
sephora_nolabel_sent['very_helpful'] = y_pred

all_with_predictions = pd.concat([sephora_labeled_sent, sephora_nolabel_sent], axis=0)   
all_with_predictions.to_pickle("all_with_predictions.p") 
helpful_reviews = all_with_predictions[all_with_predictions['very_helpful']==1] 
helpful_reviews.to_pickle("helpful_reviews.p")
