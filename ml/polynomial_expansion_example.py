from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession.builder.appName("polynomial").master("local").getOrCreate()

    df = spark.createDataFrame([
        (Vectors.dense([2.0, 1.0]),),
        (Vectors.dense([0.0, 0.0]),),
        (Vectors.dense([3.0, -1.0]),)
    ], ["features"])

    polyExpansion = PolynomialExpansion(inputCol="features", outputCol="polyFeatures", degree=3)

    polyDf = polyExpansion.transform(df)
    polyDf.show(truncate=False)

    spark.stop()