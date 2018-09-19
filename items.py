# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field
import scrapy

class SephoraItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    brand_name = scrapy.Field()
    product = scrapy.Field()
    p_id = scrapy.Field()
    p_star = scrapy.Field()
    p_category = scrapy.Field()
    p_price = scrapy.Field()
    p_num_reviews = scrapy.Field()
    p_lovecount = scrapy.Field()
    p_reviews_recommend = scrapy.Field()
    reviewer = scrapy.Field()
    r_star = scrapy.Field()
    r_eyecolor = scrapy.Field()
    r_haircolor = scrapy.Field()
    r_skintone = scrapy.Field()
    r_skintype = scrapy.Field()
    r_skinconcerns = scrapy.Field()
    r_review = scrapy.Field()
    r_helpful = scrapy.Field()
    r_nothelpful = scrapy.Field()
    r_BI = scrapy.Field()
    r_recommend = scrapy.Field()
    r_time = scrapy.Field()