from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("ChiSquareTestExample") \
        .master("local[*]") \
        .getOrCreate()

    data = [(0.0, Vectors.dense(0.5, 10.0)),
            (0.0, Vectors.dense(1.5, 20.0)),
            (1.0, Vectors.dense(1.5, 30.0)),
            (0.0, Vectors.dense(3.5, 30.0)),
            (0.0, Vectors.dense(3.5, 40.0)),
            (1.0, Vectors.dense(3.5, 40.0))]
    df = spark.createDataFrame(data, ["label", "features"])

    r = ChiSquareTest.test(df, "features", "label").head()
    print("\npValues: " + str(r.pValues))
    print("\ndegreesOfFreedom: " + str(r.degreesOfFreedom))
    print("\nstatistics: " + str(r.statistics))

    spark.stop()
