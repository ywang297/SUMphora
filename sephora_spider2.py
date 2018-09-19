from scrapy import Spider, Request
from items import SephoraItem
import re
import pandas as pd
import json
import math
import time

n_count_tot = 0

class SephoraSpider(Spider):

	name = "sephora_spider"
	#allowed_urls = ["https://www.sephora.com/ca/en", "https://api.bazaarvoice.com"]
	#start_urls = ["https://www.sephora.com/brand/list.jsp"]
	start_urls = ["https://www.sephora.com/ca/en"]

	#first is to collect all the links for all the brands
	#but this will not be used because the data is just too much. I'll just define the links

	def parse(self, response):
		#time.sleep(0.5)
		#this scrapes all of the brands
		#links = response.xpath('//a[@class="u-hoverRed u-db u-p1"]//@href').extract()
		#links = [x + "?products=all" for x in links]
		
		#brand_links = ["/fenty-beauty-rihanna", "/kiehls", "/lancome", "/estee-lauder", "/the-ordinary",
		#"/shiseido", "/sk-ii", "/clinique", "/benefit-cosmetics", "dr-jart", "/chanel", "/nars",
		#"/laneige", "/urban-decay", "/bobbi-brown"]
		brand_links = ["/clinique"]
		brand_links = [x + "?products=all&pageSize=300" for x in brand_links]

		#this scrapes only the brands inside brand_links
		links = ["https://www.sephora.com/ca/en" + link for link in brand_links]

		for url in links:
			#time.sleep(0.5)
			yield Request(url, callback=self.parse_product)

	def parse_product(self, response):
		#time.sleep(0.5)	
		dictionary = response.xpath('//script[@id="linkJSON"]/text()').extract()
		#print(dictionary[0][0])
		dictionary = re.findall('"products":\[(.*?)\]', dictionary[0])[0]
		print(dictionary)	
		product_urls = re.findall('"targetUrl":"(.*?)",', dictionary)
		product_names = re.findall('"displayName":"(.*?)"', dictionary)
		product_ids = re.findall('"productId":"(.*?)",', dictionary)
		ratings = re.findall('"rating":(.*?),', dictionary)
		brand_names = re.findall('"brandName":"(.*?)",', dictionary)
		#list_prices = re.findall('"list_price":(.*?),', dictionary)

		links2 = ["https://www.sephora.com/ca/en" + link for link in product_urls]
		#print(len(links2), len(product_names), len(product_ids), len(ratings), len(brand_names))
		if len(product_urls)!=len(ratings)!=len(brand_names):
			print('Number of products do not match with ratings')
		print(len(links2), len(product_names), len(product_ids), len(ratings), len(brand_names))	
		product_df = pd.DataFrame({'links2': links2,'product_names': product_names,'p_id': product_ids, 
			'ratings': ratings,'brand_names': brand_names})

		print (product_df.head())
		print (list(product_df.index))

		for n in list(product_df.index):
			product = product_df.loc[n, 'product_names']
			p_id = product_df.loc[n, 'p_id']
			p_star = float(product_df.loc[n, 'ratings'].strip('"'))
			brand_name = product_df.loc[n, 'brand_names']

			print (product_df.loc[n,'links2'])

			#if n>0:
				#time.sleep(20)

			yield Request(product_df.loc[n,'links2'], callback=self.parse_detail,
				meta={'product': product, 'p_id':p_id, 'p_star':p_star, 'brand_name':brand_name,})

	def parse_detail(self, response):
		#time.sleep(0.5)
		print ('parse_detail')

		product = response.meta['product']
		p_id = response.meta['p_id']
		p_star = response.meta['p_star']
		brand_name = response.meta['brand_name']

		p_category = response.xpath('//button[@class="css-1euk4ns"]/text()').extract_first() ## css-u2mtre old
		print(p_category)
		try:
			p_price = response.xpath('//div[@class="css-n8yjg7"]/text()').extract()  ## css-18suhml old
			p_price = p_price[0]
		except:
			p_price = None

		p_num_reviews = response.xpath('//span[@class="css-mmttsj"]/text()').extract() ## css-1dz7b4e  old
		p_num_reviews = p_num_reviews[0]
		p_num_reviews = p_num_reviews.replace('s', '')
		p_num_reviews = p_num_reviews.replace(' review', '')
		#p_num_reviews = p_num_reviews.replace('K', '000')
		p_num_reviews = int(p_num_reviews)
		
		p_lovecount = int(response.xpath('//span[@data-at="product_love_count"]/text()').extract()[0])

		print ('Number of reviews: {}'.format(p_num_reviews))

		#create code here that creates a list of urls for calling the reviews
		#you will use p_num_reviews, use the "{}".format technique

		#max_n = math.ceil(p_num_reviews/30)
		#low_range = [x*30 for x in list(range(0,max_n))]
		#up_range = [x*30 for x in list(range(1,max_n+1))]

		links3 = ['https://api.bazaarvoice.com/data/reviews.json?Filter=ProductId%3A' +
			p_id + '&Sort=Helpfulness%3Adesc&Limit=' + 
			'{}&Offset={}&Include=Products%2CComments&'.format(min(int(p_num_reviews), int(500)), 0) +
			'Stats=Reviews&passkey=rwbw526r2e7spptqd2qzbkp7&apiversion=5.4']

		for url in links3:
			#time.sleep(0.5)
			yield Request(url, callback=self.parse_reviews,
				meta={'product': product, 'p_id':p_id, 'p_star':p_star, 'brand_name':brand_name,
				'p_category':p_category,  'p_lovecount':p_lovecount, 'p_price':p_price})     ##'p_num_reviews':p_num_reviews,

	def parse_reviews(self, response):
		#time.sleep(0.5)
		print('parse_reviews')

		product = response.meta['product']
		p_id = response.meta['p_id']
		p_star = response.meta['p_star']
		brand_name = response.meta['brand_name']
		p_price = response.meta['p_price']
		p_category = response.meta['p_category']
		p_num_reviews = response.meta['p_num_reviews']
		p_lovecount = response.meta['p_lovecount']

		data = json.loads(response.text)
		#check keys
		#data.keys()
		p_num_reviews = data["Includes"]["Products"][p_id]["ReviewStatistics"]["TotalReviewCount"]
		p_reviews_recommend = data["Includes"]["Products"][p_id]["ReviewStatistics"]["RecommendedCount"]
		reviews = data['Results'] #this is a list
		#each element inside reviews is a dictionary
		#tmp[0].keys() will give the keys of the dictionaries inside reviews

		#create code here which arranges the data from the json dictionary into a dataframe

		n_count = 0
		global n_count_tot

		for review in reviews:
			
			try:
				reviewer = review['UserNickname']
			except:
				reviewer = None
			try:
				r_star = review['Rating']
			except:
				r_star = None

			try:
				r_eyecolor = review['ContextDataValues']['eyeColor']['Value']
			except:
				r_eyecolor = None

			try:
				r_haircolor = review['ContextDataValues']['hairColor']['Value']
			except:
				r_haircolor = None

			try:
				r_skintone = review['ContextDataValues']['skinTone']['Value']
			except:
				r_skintone = None

			try:
				r_skintype = review['ContextDataValues']['skinType']['Value']
			except:
				r_skintype = None
			try:
				r_skinconcerns = review['ContextDataValues']['skinConcerns']['Value']
			except:
				r_skinconcerns = None

			try:
				r_review = review['ReviewText']
			except:
				r_review = None
				
			try:
				r_helpful = review['TotalPositiveFeedbackCount']
			except:
				r_helpful = None
				
			try:
				r_nothelpful = review['TotalNegativeFeedbackCount']
			except:
				r_nothelpful = None
				
			try:
				r_BI = review['ContextDataValues']["beautyInsider"]['Value']
			except:
				r_BI = None
			
			try:
				r_recommend = review['IsRecommended'] ## value is true or null
				#if r_recommend == 'null':
				#	r_recommend = 0
				#if r_recommend == true:
				#	r_recommend = 1
			except:
				r_recommend = None
			
			try:
				r_time = review['LastModeratedTime']
			except:
				r_time = None

			#need to create an error handler for empty data for reviews

			print ('BRAND: {} PRODUCT: {}'.format(brand_name, product))
			print ('ID: {} STARS: {}'.format(reviewer, r_star))
			print ('='*50)

			item = SephoraItem()
			item['product'] = product
			item['p_id'] = p_id
			item['p_star'] = p_star
			item['brand_name'] = brand_name
			item['p_price'] = p_price
			item['p_category'] = p_category
			item['p_num_reviews'] = p_num_reviews 
			item['p_lovecount'] = p_lovecount
			#item['p_reviews_recommend'] = p_reviews_recommend

    		#all of these needs to be taken from the reviews list/dictionary

			item['reviewer'] = reviewer
			item['r_star'] = r_star
			item['r_eyecolor'] = r_eyecolor
			item['r_haircolor'] = r_haircolor
			item['r_skintone'] = r_skintone
			item['r_skintype'] = r_skintype
			item['r_skinconcerns'] = r_skinconcerns
			item['r_review'] = r_review
			item['r_helpful'] = r_helpful
			item['r_nothelpful'] = r_nothelpful
			item['r_BI'] = r_BI 
			item['r_recommend'] = r_recommend
			item['r_time'] = r_time

			#time.sleep(0.025)
			n_count += 1
			n_count_tot += 1

			yield item

		print ('='*50)
		print ('TOTAL NUMBER OF REVIEWS: {}'.format(int(p_num_reviews)))
		print ('NUMBER OF REVIEWS TO BE PULLED: {}'.format(len(reviews)))
		print ('ACTUAL NUMBER PULLED {}'.format(n_count))
		print ('TOTAL NUMBER PULLED {}'.format(n_count_tot))
		print ('='*50)

