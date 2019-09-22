import sys

from functools import reduce
from pyspark.sql import SparkSession

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("""
        Usage: avro_inputformat <data_file> [reader_schema_file]

        Run with example jar:
        ./bin/spark-submit --driver-class-path /path/to/example/jar \
        /path/to/examples/avro_inputformat.py <data_file> [reader_schema_file]
        Assumes you have Avro data stored in <data_file>. Reader schema can be optionally specified
        in [reader_schema_file].
        """, file=sys.stderr)
        exit(-1)

    path = sys.argv[1]

    spark = SparkSession \
        .builder \
        .appName("AvroKeyInputFormat") \
        .getOrCreate()

    sc = spark.sparkContext

    conf = None
    if len(sys.argv) == 3:
        schema_rdd = sc.textFile(sys.argv[2], 1).collect()
        conf = {"avro.schema.input.key": reduce(lambda x, y: x + y, schema_rdd)}

    avro_rdd = sc.newAPIHadoopFile(
        path,
        "org.apache.avro.mapreduce.AvroKeyInputFormat",
        "org.apache.avro.mapred.AvroKey",
        "org.apache.hadoop.io.NullWritable",
        keyConverter="org.apache.spark.examples.pythonconverters.AvroWrapperToJavaConverter",
        conf=conf)
    output = avro_rdd.map(lambda x: x[0]).collect()
    for k in output:
        print(k)

    spark.stop()
