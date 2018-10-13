# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 09:59:31 2018

@author: yishu
"""

""" Try running a simple LSTM model on Sephora reivew text """

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.layers import SpatialDropout1D

sephora = pd.read_pickle("sephora_labelled_0929.p")

max_features = 10000
tokenizer = Tokenizer(num_words=max_features, lower=True, split=' ')
tokenizer.fit_on_texts(sephora['r_review'].values)
#print(tokenizer.word_index)  # To see the dicstionary
X = tokenizer.texts_to_sequences(sephora['r_review'].values)
X = pad_sequences(X)


embed_dim = 128
lstm_out = 100
batch_size = 32

## customized metric auc to be used in Keras
from sklearn.metrics import roc_auc_score
import tensorflow as tf
def auc(y_true, y_pred):
    try:
        score = tf.py_func(lambda y_true, y_pred : roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                           [y_true, y_pred], 'float32', stateful=False, name='sklearnAUC')
        return score
    except ValueError:
        pass

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(lstm_out, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))
model.add(LSTM(lstm_out, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy']) ## change auc to 'accuracy'
print(model.summary())

#Y = sephora['very_helpful'].values
Y=pd.get_dummies(sephora['very_helpful']).values
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 29, stratify=Y)

#Here we train the Network.

model.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], batch_size =batch_size, epochs = 20, verbose = 2)
