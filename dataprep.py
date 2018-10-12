#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:02:32 2018

@author: yishu
"""

import pickle 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('wordnet')

sephora = pickle.load(open("/Users/yishu/Documents/insight/sephora.p", "rb"))

sephora.info()


def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "sephora_labeled.p")
  #  valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "sephora_nolabel.p")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

prod_review = sephora[['r_review']]

prod_review['index']=prod_review.index

documents =prod_review

documents[:5]

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


## example 

#np.random.seed(2018)

#print(WordNetLemmatizer().lemmatize('went', pos='v'))

stemmer = SnowballStemmer('english')



#def  lemmatize_stemminglemmati (text):
#    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

stop_words = list(gensim.parsing.preprocessing.STOPWORDS)
stop_words.extend(['like', 'love', 'good', 'tried', 'great'])

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result


#def preprocess(text):
#    result = []
#    for token in gensim.utils.simple_preprocess(text):
#        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#            result.append(token)
#    return result

#doc_sample = documents[documents['index'] == 4310].values[0][0]



#print('original document: ')
#words = []
#for word in documents.split(' '):
#    words.append(word)
#print(words)
#print('\n\n tokenized and lemmatized document: ')
#print(preprocess(documents))

processed_docs = documents['r_review'].map(preprocess)
#processed_docs[:10]

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
#discard the word that appeared less than 15 times and more than 50% of the time

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[0]

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
    
    
unseen_document = 'I do not think this product works well for me'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
