# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:40:48 2018

@author: yishu
"""

""" Try using word2vec on 'helpful reviews' """

import pandas as pd
import gensim 
import numpy as np
from collections import Counter

np.random.seed(29)
#PYTHONHASHSEED = 0 
""" Now if we set PYTHONHASHSEED to some number and fix workers=1 in the model definition step 
seems give non-random result, not sure if you want that, maybe it's better to have the randomness so 
we can pick a better one.. """ 
## See https://stackoverflow.com/questions/34831551/ensure-the-gensim-generate-the-same-word2vec-model-for-different-runs-on-the-sam

helpful_review =  pd.read_pickle("helpful_reviews.p")
documents = []
for review in helpful_review['r_review'].values:
    documents.append(gensim.utils.simple_preprocess(review))  
    
# build vocabulary and train model
SIZE = 50 ## The length of the word embedding vector     
model = gensim.models.Word2Vec(documents, size=SIZE, window=5, min_count=2, workers=1)
model.train(documents, total_examples=len(documents), epochs=20)    
#(5406993, 7453520) what does this output mean??

w1="oily"
model.wv.most_similar(positive=w1) ## positive (list of str, optional) – List of words that contribute positively.
                                   ## negative (list of str, optional) – List of words that contribute negatively.
                                   ## topn (int, optional) – Number of top-N similar words to return.
#[('dry', 0.74763423204422),
# ('sensitive', 0.7149749994277954),
# ('dehydrated', 0.6557960510253906),
# ('greasy', 0.6523131132125854),
# ('irritated', 0.6259891986846924),
# ('combination', 0.6034455299377441),
# ('heavy', 0.5829567909240723),
# ('drying', 0.5549484491348267),
# ('breakouts', 0.541271984577179),
# ('combo', 0.5376622080802917)]
                                                                     

w2="favorite"
model.wv.most_similar(positive=w2)
#[('favourite', 0.9276731014251709),
# ('fav', 0.7845574021339417),
# ('favorites', 0.7605363726615906),
# ('fave', 0.6953568458557129),
# ('best', 0.6271443367004395),
# ('hg', 0.6163467168807983),
# ('favourites', 0.6156744360923767),
# ('staple', 0.5484257936477661),
# ('favs', 0.5387841463088989),
# ('worst', 0.5090337991714478)]


import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE

def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0,SIZE), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.similar_by_word(word) # similar_by_word(word, topn=10, restrict_vocab=None)
                                              # Find the top-N most similar words.
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()-0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()-0.00005, y_coords.max()+0.00005)
    plt.show()
    
display_closestwords_tsnescatterplot(model, 'oily')



""" BELOW is what I try to scatterplot all points and maybe only annotate label for closest words """
# First get a list of all the words from documents
all_words = [item for sublist in documents for item in sublist]
#len(all_words)
#12768 different words in total
count = Counter(all_words)
all_words = set([word for word in all_words if count[word]>1]) ## Because in our model we set min_count=2, there is no embedding for those words only appear once.
#len(all_words)
#7747 words show up at least twice

def display_allwords(model, word):
    arr = np.empty((0,SIZE), dtype='f')
    word_labels = [word]
    close_words = model.wv.similar_by_word(word)
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
    for other_word in all_words - set(word_labels):
        arr = np.append(arr, np.array([model.wv[other_word]]), axis=0)
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords_closewords = Y[:11, 0] # coordinates of the input word itself and the 10 closes words
    y_coords_closewords = Y[:11, 1] 
    x_coords_other = Y[11:, 0]
    y_coords_other = Y[11:, 1]
    plt.scatter(x_coords_closewords, y_coords_closewords, s=1, c='r')
    plt.scatter(x_coords_other, y_coords_other, s=0.1, c='b', alpha=0.1)

    for label, x, y in zip(word_labels[:11], x_coords_closewords, y_coords_closewords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(min(list(x_coords_closewords)+list(x_coords_other))-0.00005, max(list(x_coords_closewords)+list(x_coords_other))+0.00005)
    plt.ylim(min(list(y_coords_closewords)+list(y_coords_other))-0.00005, max(list(y_coords_closewords)+list(y_coords_other))+0.00005)
    plt.show()

display_allwords(model, 'oily')

"""#######################################################################################"""
"""##################### Now Try samething with the product description data #############"""
"""#######################################################################################"""

prod_desc = pd.read_pickle("productDescriptions_raw_and_cleaned.p")    
prod_desc.dropna(subset=['p_description_clean'], inplace=True)
documents2 = []
for desc in prod_desc['p_description_clean'].values:
    documents2.append(gensim.utils.simple_preprocess(desc))  
    
# build vocabulary and train model
SIZE = 50 ## The length of the word embedding vector     
model2 = gensim.models.Word2Vec(documents2, size=SIZE, window=5, min_count=2, workers=1)
model2.train(documents2, total_examples=len(documents2), epochs=20)   

w1="oily"
model2.wv.most_similar(positive=w1)
#[('dry', 0.6836780309677124),
# ('normal', 0.6287848949432373),
# ('good', 0.5792886018753052),
# ('wick', 0.4854510426521301),
# ('coiled', 0.4727422595024109),
# ('suitable', 0.4695386290550232),
# ('ideal', 0.4511204659938812),
# ('mature', 0.4502214789390564),
# ('moisturizer', 0.4465087056159973),
# ('micronutrients', 0.44379156827926636)]

w2="favorite"
model2.wv.most_similar(positive=w2)
#[('store', 0.6065093874931335),
# ('nail', 0.6047009229660034),
# ('organize', 0.6002137660980225),
# ('final', 0.5994149446487427),
# ('magic', 0.5862186551094055),
# ('mix', 0.5832405090332031),
# ('amazing', 0.5660609602928162),
# ('unleash', 0.563684344291687),
# ('take', 0.5618799924850464),
# ('organizes', 0.5574836134910583)]