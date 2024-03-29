from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession.builder.appName("DCT").master("local").getOrCreate()

    df = spark.createDataFrame([
        (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
        (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
        (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)
    ], ["features"])

    dct = DCT(inverse=False, inputCol="features", outputCol="FeaturesDCT")

    dct.transform(df).select("featuresDCT").show(truncate=False)

    spark.stop()