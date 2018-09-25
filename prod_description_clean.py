#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 07:14:25 2018

@author: yishu
"""
import pandas as pd
import re
def textClean(text):
    if pd.notnull(text):
        if text.startswith('<div class="css-1vwy1pm">'):
            text = text[25:]
        text = text.replace('<b>', ' ') 
        text = text.replace('</b>', ' ')
        text = text.replace('<br>', '')
        text = text.replace('\r\n', ' ')
        text = text.replace('</div>', '')
        text = text.replace('<strong>', ' ')
        text = text.replace('</strong>', ' ')
        text = re.sub(r'<span[^>]*>', '', text)
        text = text.strip()
        text = text.replace('✔', '')
    return text    

productDescriptions = pd.read_csv("prod_des.csv")

productDescriptions['p_description_clean'] = productDescriptions['p_description'].apply(textClean)

productDescriptions.to_pickle("productDescriptions_raw_and_cleaned.p")