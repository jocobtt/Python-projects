import os

os.chdir('/Users/JTBras/Downloads/')
#read in data
biblerdd = spark.sparkContext.textFile('bible.txt')
biblerdd.take(5)

#perform word count normally
bibwordrdd = biblerdd.flatMap(lambda l: l.split(' '))
#map reduce function
bibmaprdd = bibwordrdd.map(lambda w: (w,1)).reduceByKey(lambda x,y: x+y)
#sort most used word in bible
sortedrdd = bibmaprdd.sortBy(lambda a: a[1],ascending=False)
#top 20 words
sortedrdd.take(20)

#import libraries
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
#set stop_words value
stop_words = set(stopwords.words("english"))

#stem, tokenize, and remove stopwords using nltk
stemmer = PorterStemmer()
tokedd = biblerdd.flatMap(lambda x: TreebankWordTokenizer().tokenize(x.lower()))
stemrdd = tokedd.map(lambda w: stemmer.stem(w))
stoprdd = stemrdd.filter(lambda w: w not in stop_words)

#find most used words in bible
wocutrdd = stoprdd.map(lambda w: (w,1)).reduceByKey(lambda x,y: x+y)
sortrdd = wocutrdd.sortBy(lambda a: a[1],ascending=False)
#get top 20 most used words
sortrdd(20)
