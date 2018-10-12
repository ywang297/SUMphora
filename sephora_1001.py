# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:04:31 2018

@author: yishu
"""

import pandas as pd

label = pd.read_pickle("sephora_labelled_0929.p")

label_no = pd.read_pickle("sephora_nolabel_0929.p")

all_reviews = pd.concat([label, label_no])

review_counts_by_product = pd.DataFrame(all_reviews['p_id'].value_counts())
# 916 different products

review_counts_by_product.rename(columns={'p_id':'num_reviews'}, inplace=True)

len(review_counts_by_product[review_counts_by_product['num_reviews']>20])
# 372

len(review_counts_by_product[review_counts_by_product['num_reviews']>10])
# 522

sum(review_counts_by_product[review_counts_by_product['num_reviews']>20]['num_reviews'])
# 19386

sum(review_counts_by_product[review_counts_by_product['num_reviews']>10]['num_reviews'])
# 21716

""" Find the most important features for the random forest model (with certain hyperparameter) step by step """
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import scipy

np.random.seed(29)

sephora_labeled = pd.read_pickle("sephora_labelled_0929.p")
sephora_nolabel = pd.read_pickle("sephora_nolabel_0929.p")

sephora_labeled.dropna(subset=['p_category'], inplace=True)

def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    pattern = re.compile('\W')
    string = re.sub(pattern, ' ', str(message))
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

from sklearn.feature_extraction.text import TfidfVectorizer
# preprocess the 'r_review' text input into tf_idf vectors
tfidf_vec = TfidfVectorizer(max_df=0.5, max_features=5000, ngram_range=(1,2), norm='l2', stop_words='english', smooth_idf=True, sublinear_tf=True, use_idf=True)
tfidf_vec.fit(sephora_labeled['r_review'])
review_vec = tfidf_vec.transform(sephora_labeled['r_review'])
# 13429x5000 sparse matrix
tfidf_vec.get_feature_names()
# tfidf_vec.get_feature_names()[0], ... tfidf_vec.get_feature_names()[4999] gives all the 5000 features

# Prepreocess the categorical varaible 'p_category'
enc = OneHotEncoder()
enc.fit(sephora_labeled['p_category'].values.reshape(-1,1))
p_category = enc.transform(sephora_labeled['p_category'].values.reshape(-1,1))
enc.get_feature_names()
#array(['x0_Bath & Body', 'x0_Fragrance', 'x0_Hair', 'x0_Makeup',
#       'x0_Skincare'], dtype=object)

# Scale the numerical variables 'review_length', 'neg', 'pos', 'r_star'
scale = StandardScaler()
numerical_feature_scaled = scale.fit(sephora_labeled[['review_length', 'neg', 'pos', 'r_star']].values)

# Put all variables together as input xs, and ys is the label
xs = scipy.sparse.hstack((review_vec, p_category, sephora_labeled[['review_length', 'neg', 'pos', 'r_star']].values.astype(float)))
ys = sephora_labeled['very_helpful'].values

# Random forest classification
clf = RandomForestClassifier(random_state=29, n_estimators=100, 
                             min_samples_leaf=1, max_depth=4, bootstrap=True)
clf.fit(xs, ys)
clf.feature_importances_

# Get a list of all features, combine it with feature_importances to get a dataframe to look at
feature_list = list(tfidf_vec.get_feature_names())+list(enc.get_feature_names())+['review_length', 'neg', 'pos', 'r_star']
feature_importances = pd.DataFrame({'feature': feature_list, 'importance': clf.feature_importances_})
