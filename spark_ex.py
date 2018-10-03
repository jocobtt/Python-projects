import pyspark.sparkContext as sc
import pyspark
count = spark.sparkContext.parallelize(range(0,1024))
count
count.first()
count.take(5)
#estimate pi
NUM_SAMPLES = 65536
samplerdd = spark.sparkContext.parallelize(range(0,NUM_SAMPLES))
import random
def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1
evenrdd = samplerdd.filter(lambda x: x%2 == 0)
samplerdd.count()
evenrdd.count()
evenrdd.take(5)
insiderdd = samplerdd.filter(inside)
count = insiderdd.count()
print("PI is roughly %f" % (4.0 * count / NUM_SAMPLES))
#take larger sample to estimate pi more accurately
NUM_SAMPLES = 5000000
count = spark.sparkContext.parallelize(range(0,NUM_SAMPLES)).filter(inside).count()
count
print("PI is roughly %f" % (4.0 * count / NUM_SAMPLES))
#cleans up context and shuts down correctly
spark.stop()
quit()

#new dataset- bible dataset
linesrdd = spark.sparkContext.textFile('bible.txt') #textFile returns lines of text as list rdd
linesrdd.count()
linesrdd.first()
linesrdd.take(5)
#count words in the file- flatmap it
wordsrdd = biblerdd.flatMap(lambda l: l.split(' '))
wordsrdd.take(5)
#how many words in bible
wordsrdd.count()
#map & reduce
wcountlistrdd = wordsrdd.map(lambda w: (w,1)).reduceByKey(lambda x,y: x+y)
wcountlistrdd.take(5)
#find most used word in bible
sortedrdd = wcountlistrdd.sortBy(lambda a: a[1],ascending=False)
#sortedrdd = wcountlistrdd.sortBy(lambda a: -a[1],ascending=False)

sortedrdd.take(5)
sortedrdd.take
sorted = sortedrdd.collect()
#gives you whole list
sorted
#remove stopwords and tokenize
spark.stop()
#important thing to run with pyspark libarary
sc.setLogLevel('WARN')


#bible wordcount assignment
#remove stopwords and seperate according to word
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))

#read in data
biblerdd = spark.sparkContext.textFile('/Users/JTBras/Downloads/bible.txt')
biblerdd.take(5)


#tokenize
biblerdd.fillna("").map(nltk.word_tokenize)
#lowercase
def word_clean(x):
  return re.sub("[^a-zA-Z0-9\s]+","", x).lower().strip()

def numcl(x):
  return re.sub('[0-9]+','', x)
#lowercase
tokerdd = biblerdd.map(lambda x : word_clean(x))

cleanrdd = tokerdd.map(lambda x : numcl(x))
#flatmap/split
wordsrdd = cleanrdd.flatMap(lambda l: l.split(' '))
#map & reduce
wordrdd = wordsrdd.map(lambda w: (w,1)).reduceByKey(lambda x,y: x+y)
#remove stopwords
keyvrdd = wordrdd.filter(lambda x : x[0] not in stop_words)
#remove blank spaces
blanksrdd = keyvrdd.filter(lambda x : x[0] not in " ")



wordrdd.take(5)
#find most used word in bible
sortedrdd = wordrdd.sortBy(lambda a: a[1],ascending=False)

#netflix problem:
#login
ssh brasw@ssh.fsl.byu.edu

cd spark-2.0.0/bin
./pyspark

from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession

def splitKeyFromValue(line):
    array = line.split()
    return ((array[1],array[2],{array[0]}))

def userInSet(user,set):
    userIdSet = {user}
    return set.issuperset(userIdSet)
def spreadByUser(tuple):
    results = []
    for user in tuple[1]:
        results.append((user,1))
    return results

net_rdds = spark.sparkContext.textFile('/fslgroup/fslg_hadoop/netflix_data.txt')

movieRanks = net_rdds.map(lambda net_rdds: splitKeyFromValue(net_rdds)) \
    .reduceByKey(lambda a,b : a.union(b)) \
    .filter(lambda tuple: userInSet(userId,tuple[1]))
movieRanks.take(5)

similarUsers = movieRanks \
    .flatMap(lambda rank: spreadByUser(rank)) \
    .filter(lambda user: user[0] != userId)
similarUsers.take(5)
counts = similarUsers.reduceByKey(lambda a, b: a+b)
ordered = counts.sortBy(lambda a: a[1], ascending=False)
ordered.take(10)




import pyspark.sql.functions as func
from pyspark.sql.functions import desc
#map reduce
#movies that user 1488844 ranked
user_movies = df_net.filter(df_net['ID']==1488844)
user_movies.take(50)

df_net.filter(df_net[''])

val = df_net.agg(func.count('MovieID'==user_movies['MovieID'])).alias('counts')
    .take(10)

#mapreduce- count users that gave out fives
fivesrdd = net_tabx.filter(lambda x: (x[2]==5)) \
    .map(lambda x: x[0],x[1]) \
    .mapValues(lambda x: (x,1)) \
    .reduceByKey(lambda x[0], y[0]: (x+y)) \
    .sortBy(lambda a: a[1],ascending=False) \
    .take(20)

#reduce to same movieid and count
movieidrdd = fivesrdd.map(lambda x: (x[:,1]=6789)).reduceByKey(lambda x,y:x+y)
sortmovierdd = movierdd.sortBy(lambda a: a[1],ascending=False)

#top 20
sortmovierdd.take(20)

#convert to dataframe
df_net = net_tabx.toDF(['ID','MovieID','Rating'])
#print column names in df
df_net.printSchema()
#explode dataset
explodeDF = df_net.selectExpr('e.ID','e.MovieID','e.Rating')
explodeDF.take(5)
#mapreduce
keyvrdd = df_net.
