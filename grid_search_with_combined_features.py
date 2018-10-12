# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 00:49:05 2018

@author: Yishu
"""

""" Need to run with the myenv environment where we have scikitlearn version 0.20 which contain sklearn.compose """


""""grid search for logistic regression """"

import pickle
import collections,re
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    pattern = re.compile('\W')
    string = re.sub(pattern, '', str(message))
    message = string.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

sephora_labeled_sent = pd.read_pickle("/Users/yishu/Documents/insight/sephora_labeled_sent_new_new.p")

sephora_nolabel_sent = pd.read_pickle("/Users/yishu/Documents/insight/sephora_nolabel_sent_new_new.p")


np.random.seed(29)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sephora_labeled_sent[['r_review', 'review_length', 'neg', 'pos', 'r_star']], 
                                                    sephora_labeled_sent['very_helpful'], 
                                                    test_size=0.2, random_state=29)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

text_transformer = Pipeline(steps=[('vect', CountVectorizer(analyzer=split_into_lemmas, max_df=0.5)),
                                   ('tfidf', TfidfTransformer())])


preprocessor = ColumnTransformer(transformers=[
        ('review', text_transformer, 'r_review'),
        ('numerical', StandardScaler(), ['review_length', 'neg', 'pos', 'r_star'])])

lr_pipeline = Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(random_state=29))])

lr_parameters = {
    'preprocessor__review__vect__max_df': [0.5],
    'preprocessor__review__vect__max_features': [1000,5000],
    'preprocessor__review__vect__ngram_range': [(1, 2)],
    'preprocessor__review__tfidf__use_idf': [True],
    'preprocessor__review__tfidf__norm': ['l2'],
    'preprocessor__review__tfidf__smooth_idf': [True],
    'preprocessor__review__tfidf__sublinear_tf': [True],
    'clf__C': [1], 
    'clf__penalty': ['l2','l1'],
    'clf__tol': [0.001]
}

lr_grid = GridSearchCV(lr_pipeline, lr_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
lr_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 232.34556698799133
lr_grid.best_score_
#0.6210483196078691
lr_grid.best_params_

sorted(lr_grid.cv_results_.keys())



predicted = lr_grid.predict(x_test)

from sklearn import metrics
print(metrics.classification_report(y_test, predicted))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted)
plt.matshow(cm)



""""grid search for SVM """"

from sklearn.linear_model import SGDClassifier


text_transformer = Pipeline(steps=[('vect', CountVectorizer(analyzer=split_into_lemmas)),
                                   ('tfidf', TfidfTransformer())])


preprocessor = ColumnTransformer(transformers=[
        ('review', text_transformer, 'r_review'),
        ('numerical', StandardScaler(), ['review_length', 'neg', 'pos', 'r_star'])])

svm_pipeline = Pipeline([('preprocessor', preprocessor), ('clf2', SGDClassifier(loss='modified_huber',
                                             random_state=1))])

svm_parameters = {
    'preprocessor__review__vect__max_df': [0.5],
    'preprocessor__review__vect__max_features': [1000,5000,10000],
    'preprocessor__review__vect__ngram_range': [(1, 2)],
    'preprocessor__review__tfidf__use_idf': [True],
    'preprocessor__review__tfidf__norm': ['l2'],
    'preprocessor__review__tfidf__smooth_idf': [True],
    'preprocessor__review__tfidf__sublinear_tf': [True],
    'clf2__penalty': ['l2','l1'],
    'clf2__tol': [0.0001]
}

svm_grid = GridSearchCV(svm_pipeline, svm_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)


t0=time.time()
svm_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 232.34556698799133
svm_grid.best_score_
#0.6210483196078691
svm_grid.best_params_



#
#clf2_svm = Pipeline([('vect', CountVectorizer(analyzer=split_into_lemmas, max_df=0.5)),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', SGDClassifier(loss='modified_huber', penalty='l2',
#                                            alpha=1e-4, n_iter=5, random_state=1)),])
predicted = clf_svm.predict(X_test)

np.mean(predicted==y_test)


""""grid search for random forest """"


rf_pipeline = Pipeline([('preprocessor', preprocessor), ('clf3', RandomForestClassifier(
                                             random_state=1))])


#forest = RandomForestClassifier(n_estimators = 10, max_depth=5)
#forest = forest.fit(sephora_labeled_tfidf, sephora_labeled_sent["bi_helpful"] )

rf_parameters = {
    'preprocessor__review__vect__max_df': [0.5],
    'preprocessor__review__vect__max_features': [10000],
    'preprocessor__review__vect__ngram_range': [(1, 2)],  # unigrams or bigrams
    'preprocessor__review__tfidf__use_idf': [True],
    'preprocessor__review__tfidf__norm': ['l2'],
    'preprocessor__review__tfidf__smooth_idf': [True],
    'preprocessor__review__tfidf__sublinear_tf': [True],
    'clf3__n_estimators': [100],
    'clf3__max_depth': [5],
    'clf3__min_samples_split': [5],
    'clf3__min_samples_leaf': [1], 
    'clf3__bootstrap': [True]
}   

rf_grid = GridSearchCV(rf_pipeline, rf_parameters, cv=5, n_jobs=1, scoring='roc_auc', verbose=1)

import time
t0=time.time()
rf_grid.fit(X=x_train, y=y_train)
t1=time.time()
print('time:', t1-t0)
#time: 5213.134449005127
rf_grid.best_score_
# 0.6018439947318756, std: 0.030261
rf_grid.best_params_
#{'clf__bootstrap': True,
# 'clf__max_depth': 5,
# 'clf__min_samples_leaf': 2,
# 'clf__min_samples_split': 5,
# 'clf__n_estimators': 50,
# 'preprocessor__review__tfidf__norm': 'l2',
# 'preprocessor__review__tfidf__smooth_idf': True,
# 'preprocessor__review__tfidf__sublinear_tf': True,
# 'preprocessor__review__tfidf__use_idf': True,
# 'preprocessor__review__vect__max_df': 0.5,
# 'preprocessor__review__vect__max_features': 5000,
# 'preprocessor__review__vect__ngram_range': (1, 2)}
rf_cv_results=pd.DataFrame(rf_grid.cv_results_)
#obeservations:
#n_estimators: 50 definitely better than 20, could try more
#max_depths: 5 better than 4
#min_samples_leaf: 2 better than 10
#min_samples_split: does not matter (at least for the choice of these other parameters)    



#### making a roc plot #####

probs = lr_grid.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)


probs_svm = svm_grid.predict_proba(x_test)
preds_svm = probs_svm[:,1]
fpr_svm, tpr_svm, threshold = metrics.roc_curve(y_test, preds_svm)
roc_auc_svm= metrics.auc(fpr_svm, tpr_svm)


probs_rf = rf_grid.predict_proba(x_test)
preds_rf = probs_rf[:,1]
fpr_rf, tpr_rf, threshold = metrics.roc_curve(y_test, preds_rf)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',  label = 'AUC_logistic = %0.2f' % roc_auc)
plt.plot(fpr_svm, tpr_svm, 'c',label= 'AUC_SVM = %0.2f' % roc_auc_svm) 
plt.plot(fpr_rf, tpr_rf, 'y',label= 'AUC_randomForest = %0.2f' % roc_auc_rf)         
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True positive rate (TPR)')
plt.xlabel('False positive rate (FPR)')
plt.show()



sephora_nolabel_sent = pd.read_pickle("/Users/yishu/Documents/insight/sephora_nolabel_0929.p")


nolabel_test = sephora_nolabel_sent[["r_review", 'review_length', 'neg', 'pos', 'r_star','p_category']]

nolabel_predict =svm_grid.predict(nolabel_test)

nolabel_predict.mean()*10122


sephora_labeled_sent.very_helpful.mean()*13429+8209.0




