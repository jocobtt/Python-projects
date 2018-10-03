#Results
'''
Group: Jacob Braswell, Tyki Wada, Joshua Robinson, Isaac Whittaker
users and numbers output
[(u'2602249', 814), (u'322009', 775), (u'16272', 724), (u'1314869', 707), (u'1559083', 687),
(u'1139570', 674), (u'901520', 670), (u'1110156', 666), (u'243612', 666), (u'184705', 658)]

'''

#netflix spark assignment
import sys
from operator import add
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Netflix user').getOrCreate()

user = '1488844'
file = '/fslhome/brasw/netflix_data.txt'
output = '/fslhome/brasw/compute'
my_usr_rdd = spark.read.text(file).rdd.map(lambda r: r[0])
users_ratingsame = my_usr_rdd.map(lambda l: l.split('\t')) \
    .map(lambda x: (x[1] + ‘:’ + x[2], [x[0]])) \
    .reduceByKey(add) \
    .filter(lambda x: user in r[1]) \
    .flatMap(lambda x: [(y, 1) for y in x[1]]) \
    .filter(lambda y: u[y] != user) \
    .reduceByKey(add) \
    .sortBy(lambda x: x[1], ascending=False)
print str(users_ratingsame.take(10))
spark.stop()
