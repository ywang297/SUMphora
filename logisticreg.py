#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:43:53 2018

@author: yishu
"""

## train the helpfulness model on the labeled dataset 


## random forest
import pickle
import collections,re
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt




def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    pattern = re.compile('\W')
    string = re.sub(pattern, '', str(message))
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

sephora_labeled_sent = pickle.load(open("/Users/yishu/Documents/insight/sephora_labeled_sent.p", "rb"))

sephora_nolabel_sent = pickle.load(open("/Users/yishu/Documents/insight/sephora_nolabel_sent.p", "rb"))

sephora_labeled_sent.bi_helpful.describe()

sephora_nolabel_sent.bi_helpful.describe()

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(sephora_labeled_sent['r_review'])
sephora_labeled_bow = bow_transformer.transform(sephora_labeled_sent.r_review)


#stopwords = set(STOPWORDS)
#stopwords.add('like')
#stopwords.add('love')
#stopwords.add('tried')
#stopwords.add('good')
#stopwords.add('great')
#stopwords.add('make')

tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True).fit(sephora_labeled_bow)
sephora_labeled_tfidf = tfidf_transformer.transform(sephora_labeled_bow)

sephora_nolabel_sent = pickle.load(open("/Users/yishu/Documents/insight/sephora_nolabel_sent.p", "rb"))

sephora_nolabel_sent_bow = bow_transformer.transform(sephora_nolabel_sent.r_review)

sephora_nolabel_sent_tfidf = tfidf_transformer.transform(sephora_nolabel_sent_bow)

forest = RandomForestClassifier(n_estimators = 10, max_depth=5)
forest = forest.fit(sephora_labeled_tfidf, sephora_labeled_sent["bi_helpful"] )

result = forest.predict(sephora_nolabel_sent_tfidf)

output = pd.DataFrame( data={"r_review":sephora_nolabel_sent["r_review"], "Prediction":result} )

output.describe()

output.to_csv( "Bag_of_Words_randomF.csv", index=False, sep="," )



## SVM  not modified yet 

from sklearn.cross_validation import train_test_split

#df_train = pd.read_csv('ml_dataset_train.csv', sep=',', header='infer', low_memory=False)
#df_train.dropna(how="any", inplace=True)

train_set, test_set = train_test_split(sephora_labeled_sent, test_size=0.33, random_state=29)


X_test = test_set["r_review", "review_length", "neg", "pos", "neu"]

X_test = test_set["r_review"]
y_test = test_set['bi_helpful']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
clf_svm = Pipeline([('vect', CountVectorizer(max_df=0.5)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-4, n_iter=5, random_state=1)),])
_ = clf_svm.fit(train_set['r_review'], train_set['bi_helpful'])
predicted = clf_svm.predict(X_test)

np.mean(predicted==y_test)

from sklearn import metrics
print(metrics.classification_report(y_test, predicted))

from sklearn import metrics
metrics.confusion_matrix(y_test, predicted)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted)
plt.matshow(cm)

nolabel_test = sephora_nolabel_sent["r_review"]

nolabel_predict =clf_svm.predict(nolabel_test)



probs = clf_svm.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




