import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, NoAlertPresentException
from selenium.webdriver.support.ui import WebDriverWait
import sys
import pandas as pd
from bs4 import BeautifulSoup
import re, os
import time

class SelScrape(object):
    def __init__(self):
        # set up driver w/ the options for driver
        self.option = webdriver.ChromeOptions()
        self.option.add_argument(" - icognito")
        self.option.add_argument('--headless')
        self.option.add_argument('--no-sandbox')
        # set up chrome driver
        self.driver = webdriver.Chrome(executable_path='/Users/JTBras/Downloads/chromedriver', options=option)

        # set base url for where to start search
        self.base_url = 'https://twitter.com/search?q='

    # function to pull tweets
    def pull_tweets(self, query, num):
        try:
            self.driver(self.base_url+query)
            time.sleep(3)
            # find based on tag name== body
            bod = driver.find_element_by_tag_name('body')
            # scroll down through # of tweets we define
            for _ in range(num):
                bod.send_keys(Keys.PAGE_DOWN)
                time.sleep(1)
            # pull out what we pulled in the above for loop statement - can either do it by text or by timeline
            tweets = self.driver.find_elements_by_class_name('tweet-text')
            hreff = self.driver.find_elements_by_class_name('twitter-timeline-link')
            # return pandas dataframe w/ our tweets
            # for link for scott have it pull the href
            return pd.DataFrame({'tweets': [tweets.text for tweet_text in tweets], 'link': [hreff.text for link in hreff]})
            # error statement
        except:
            print('something is wrong- an error occured while pulling tweets for you')

# to use the class that was built - input number of tweets you want to scrape and what you want it to search
scrape = SelScrape()
tweets_df = SelScrape.pull_tweets('npr', 30)
