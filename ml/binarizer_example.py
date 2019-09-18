from pyspark.ml.feature import Binarizer
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("binarizer").master("local").getOrCreate()

    dataFrame = spark.createDataFrame([
        (0, 0.1), (1, 0.8), (2, 0.3)
    ], ["id", "feature"])

    binaizer = Binarizer(inputCol="feature", outputCol="binarizer", threshold=0.5)

    binarizerDataFrame = binaizer.transform(dataFrame)

    binarizerDataFrame.show()

    spark.stop()