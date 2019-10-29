from pyspark import SparkContext, SparkConf

sc = SparkContext(conf=SparkConf())
files = sc.wholeTextFiles("tmpdata")
print(files.collect())

textFile = sc.textFile("file:///usr/local/spark/mycode/wordcount/word.txt")
wordCount = textFile.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(
    lambda a, b: a + b).collect()
