import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile(sys.argv[1])
words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))

words_alpha = words.filter(lambda w: w and w[0].isalpha())
words_lower = words_alpha.map(lambda w: w.lower())
words_unique = words_lower.distinct()

pairs = words_unique.map(lambda w: (w[0], 1))
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
counts_dict = dict(counts.collect())

for i in range(ord('a'), ord('z') + 1):
    char = chr(i)
    print(f"{char}\t{counts_dict.get(char, 0)}")

sc.stop()
