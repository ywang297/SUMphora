# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 05:51:12 2018

@author: Tao
"""

import pandas as pd

jimmy = pd.read_csv("jimmy-choo_reviews.csv")

jimmy = jimmy[jimmy.index.values %2 !=0]

kiehls = pd.read_csv("kiehls_reviews.csv")
lancome = pd.read_csv("lancome_reviews.csv")
shiseido = pd.read_csv("shiseido_reviews.csv")
dior = pd.read_csv("dior_reviews.csv")
fresh = pd.read_csv("fresh_reviews.csv")
clarins = pd.read_csv("clarins_reviews.csv")
clinique = pd.read_csv("clinique_reviews.csv")

sephora = pd.concat([kiehls, lancome, shiseido, dior, fresh, clarins, clinique])



sephora.to_csv("sephora.csv", index=False, sep=',')
import pickle
pickle.dump(sephora, open("sephora.p", "wb"))
