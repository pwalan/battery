from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.set("master", "local[4]")
sc = SparkContext(conf=conf)

# textFile = sc.textFile("file:///Users/alanp/Projects/201806_高比功率电池项目/数据/beili/sample.txt")
textFile = sc.textFile("file:///Users/alanp/Projects/201806_高比功率电池项目/数据/beili/1ca6920e828449cda6e4e1bb12f9fe0b")

fields = textFile.map(lambda line: line.split(","))

print("总条数", fields.map(lambda fields: fields[0]).count())
print("故障数", fields.map(lambda fields: fields[51]).filter(lambda x: len(x.split(",")) == 2).filter(
    lambda x: x.split(":")[1] != '0').count())
print("报警数", fields.map(lambda fields: fields[57]).filter(lambda x: len(x.split(",")) == 2).filter(
    lambda x: x.split(":")[1] != '0').count())
