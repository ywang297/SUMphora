# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 05:51:12 2018

@author: Yishu
"""

import pandas as pd
import pickle 
import numpy as np


jimmy = pd.read_csv("jimmy-choo_reviews.csv")

bio = pd.read_csv("biotherm_reviews_290.csv")

bbrown = pd.read_csv("bobbi-brown_reviews_1327.csv")

ct = pd.read_csv("charlotte_tilbury_reviews_82.csv")

evelom = pd.read_csv("eve-lom_reviews_378.csv")

mac = pd.read_csv("mac-cosmetics_reviews_113.csv")

muf = pd.read_csv("make-up-for-ever_reviews_2252.csv")

sc = pd.read_csv("sephora-collection_reviews_5084.csv")

ysl = pd.read_csv("yves-saint-laurent_reviews_1244.csv")

kiehls = pd.read_csv("kiehls_reviews.csv")
lancome = pd.read_csv("lancome_reviews.csv")
shiseido = pd.read_csv("shiseido_reviews.csv")
dior = pd.read_csv("dior_reviews.csv")
fresh = pd.read_csv("fresh_reviews.csv")
clarins = pd.read_csv("clarins_reviews.csv")
clinique = pd.read_csv("clinique_reviews.csv")

sephora_old = pd.concat([kiehls, lancome, shiseido, dior, fresh, clarins, clinique])


shu = pickle.load(open("/Users/yishu/Documents/insight/shu_uemura.p", "rb"))
philosophy = pickle.load(open("/Users/yishu/Documents/insight/philosophy.p", "rb"))



sephora =pd.concat([sephora_old, shu, philosophy, bio, bbrown, ct, evelom, mac,
                    muf, sc, ysl])

sephora.loc[sephora['p_category']== 'Makeup Palettes', 'product']


sephora['brand_name'].value_counts()

sephora['p_category'].value_counts()

skincare_list =['Skincare', 'Hidden Category']


sephora.loc[sephora['p_category'].isin(skincare_list),'p_category'].value_counts().sum()

sephora.loc[sephora['p_category'].isin(skincare_list), 'p_category'] = 'Skincare'


makeup_list =['Makeup','Makeup Palettes']

sephora.loc[sephora['p_category'].isin(makeup_list),'p_category'].value_counts().sum()

sephora.loc[sephora['p_category'].isin(makeup_list), 'p_category'] = 'Makeup'


remove_list = ['Value & Gift Sets', 'Makeup & Travel Cases', 'Men', 'Nail', 'Self Tanners',
              'Tools & Brushes', 'Highlighter Brush', 'Foundation Brush', 
              'Candles & Home Scents','Makeup Brushes & Applicators', 'Brush Sets & Accessories',
             'Powder Brush', 'Crease Shadow Brush', 'Blush Brush','Mini Size']

sephora.loc[sephora['p_category'].isin(remove_list),'p_category'].value_counts().sum()

#24392-819
sephora = sephora[~sephora['p_category'].isin(remove_list)]

sephora.to_csv("sephora.csv", index=False, sep=',')

sephora.to_pickle("sephora.p")


sephora["r_helpful"] = sephora.r_helpful.astype(float)
sephora["r_nothelpful"] = sephora.r_nothelpful.astype(float)

sephora['helpfulrate']= np.where((sephora['r_helpful']+sephora['r_nothelpful'])==0,-1, sephora['r_helpful']/(sephora['r_helpful']+sephora['r_nothelpful'])) 

sephora['numvote'] = sephora["r_helpful"]+sephora["r_nothelpful"] 

sephora_labeled =sephora[sephora.numvote>0]

sephora_nolabel =sephora[sephora.numvote==0]


sephora_labeled.numvote.describe()

sephora_labeled.to_pickle("sephora_labeled.p")

sephora_nolabel.to_pickle("sephora_nolabel.p")



