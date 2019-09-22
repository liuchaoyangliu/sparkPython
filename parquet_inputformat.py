import sys

from pyspark.sql import SparkSession

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("""
        Usage: parquet_inputformat.py <data_file>

        Run with example jar:
        ./bin/spark-submit --driver-class-path /path/to/example/jar \\
                /path/to/examples/parquet_inputformat.py <data_file>
        Assumes you have Parquet data stored in <data_file>.
        """, file=sys.stderr)
        exit(-1)

    path = sys.argv[1]

    spark = SparkSession \
        .builder \
        .appName("ParquetInputFormat") \
        .getOrCreate()

    sc = spark.sparkContext

    parquet_rdd = sc.newAPIHadoopFile(
        path,
        'org.apache.parquet.avro.AvroParquetInputFormat',
        'java.lang.Void',
        'org.apache.avro.generic.IndexedRecord',
        valueConverter='org.apache.spark.examples.pythonconverters.IndexedRecordToJavaConverter')
    output = parquet_rdd.map(lambda x: x[1]).collect()
    for k in output:
        print(k)

    spark.stop()
