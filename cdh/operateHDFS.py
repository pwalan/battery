from hdfs import *

client = Client("http://namenode:9870")

print(client.list("/data"))

