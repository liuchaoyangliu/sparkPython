import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.flume import FlumeUtils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: flume_wordcount.py <hostname> <port>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="PythonStreamingFlumeWordCount")
    ssc = StreamingContext(sc, 1)

    hostname, port = sys.argv[1:]
    kvs = FlumeUtils.createStream(ssc, hostname, int(port))
    lines = kvs.map(lambda x: x[1])
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()
