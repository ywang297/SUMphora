#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 06:41:22 2018

@author: yishu
"""
import pickle 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sephora= pd.read_pickle("/Users/yishu/Documents/insight/sephora.p")

sephora["r_helpful"] = sephora.r_helpful.astype(float)
sephora["r_nothelpful"] = sephora.r_nothelpful.astype(float)

sephora['helpfulrate']= np.where((sephora['r_helpful']+sephora['r_nothelpful'])==0,-1, sephora['r_helpful']/(sephora['r_helpful']+sephora['r_nothelpful'])) 

sephora['helpful_label'] = np.where(sephora['numvote']>0, 1, 0 )

sephora.helpful_label.sum()

sephora.helpful_label.astype(str)

sephora.describe().transpose()

plt.hist(sephora.helpfulrate[sephora.helpfulrate>-0.2])

sephora.helpfulrate[sephora.helpfulrate>-0.2].count()

sephora.helpfulrate[sephora.helpfulrate==-1].count()


sephora.r_helpful[sephora.helpfulrate != -1].describe()

sephora.r_nothelpful[sephora.helpfulrate != -1].describe()


plt.figure(figsize=(10,10))
sns.boxplot(x = 'pos', y = 'p_star', data = sephora_sent)
 

plt.figure(figsize=(4,3))
figsent = sns.boxplot(x = 'r_recommend', y = 'compound', data = sephora_sent)
figsent.set(xlabel='whether recommend this product', ylabel='sentiment of reviews')
plt.show()



figstar = sns.boxplot(x = 'r_recommend', y = 'p_star', data = sephora_sent)
figstar.set(xlabel='whether recommend this product', ylabel='rating of the product')
plt.show()


figprice = sns.boxplot(x = 'r_recommend', y = 'p_price', data = sephora_sent)
figstar.set(xlabel='whether recommend this product', ylabel='rating of the product')
plt.show()







labels = 'helpful', 'not helpful' , 'no label'
sizes = [7235, 6194, 10122]
colors = ['lightskyblue', 'yellowgreen', 'orange']
#explode = (0.1, 0, 0)  # explode 1st slice
 
#fig = plt.figure(1, figsize=(6,6))
#ax = fig.add_subplot(111)

# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=140, textprops={'fontsize': 20})
 
plt.axis('equal')

#patches, texts, autotexts = ax.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%')
#texts[0].set_fontsize(10)
#texts[1].set_fontsize(10)
#texts[2].set_fontsize(10)
plt.show()


ax1 = plt.hist(sephora['helpful_label'], color = "red", alpha = 0.5, label = "helpful votes")
ax1 = plt.hist(nrd['r_recommend'], color = "blue", alpha = 0.5, label = "Not Recommended")
ax1 = plt.title("Recommended Items in each Division")
ax1 = plt.legend()





count_reviewer = pd.value_counts(sephora['reviewer'])

pd.value_counts(sephora['p_category'])

pd.value_counts(sephora_old['brand_name'])

sephora['brand_name'].value_counts()

pd.value_counts(sephora['r_recommend'])

pd.value_counts(sephora['r_skinconcerns'])

pd.value_counts(sephora['r_skintype'])


sephora['r_star'].describe()

sephora['r_helpful'].describe()

sephora['r_nothelpful'].describe()


#Masked wordcloud
#================
#Using a mask you can generate wordclouds in arbitrary shapes.


from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud, STOPWORDS

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

sephora = pickle.load(open("/Users/yishu/Documents/insight/sephora_new.p", "rb"))


prod_review = sephora[['r_review']]

prod_review['index']=prod_review.index

documents =prod_review


# Read the whole text.
#text = open(path.join(d, 'run.txt')).read()
#
#text = sephora['r_review']
#
#text = documents.r_review.apply(str)

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
alice_mask = np.array(Image.open(path.join(d, "alice_mask.png")))

stopwords = set(STOPWORDS)
stopwords.add('like')
stopwords.add('love')
stopwords.add('tried')
stopwords.add('good')
stopwords.add('great')
stopwords.add('make')

#wc = WordCloud.generate_from_frequencies(background_color="white", max_words=2000, mask=alice_mask,
#               stopwords=stopwords)

# generate word cloud, this new command change the series object to string
wordcloud2 = WordCloud(max_words=2000, 
               stopwords=stopwords).generate(' '.join(documents['r_review']))




# store to file
wordcloud2.to_file(path.join(d, "alice_mask.png"))

# show
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.figure()
#plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
#plt.axis("off")
#plt.show()


import pyLDAvis.gensim
pyLDAvis.enable_notebook()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

pyLDAvis.gensim.prepare(model, corpus, dictionary)


plt.style.use('ggplot')

