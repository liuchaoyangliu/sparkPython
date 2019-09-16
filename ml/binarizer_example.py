from pyspark.sql import SparkSession
from pyspark.ml.feature import Binarizer

if __name__ == "__main__":

    spark = SparkSession.builder.appName("BinarizerExample").master("local").getOrCreate()

    continuousDataFrame = spark.createDataFrame([
        (0, 0.1),
        (1, 0.8),
        (2, 0.2)
    ], ["id", "feature"])

    binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

    binarizedDataFrame = binarizer.transform(continuousDataFrame)

    print("具有阈值的二进制化器输出 = %f" % binarizer.getThreshold())
    binarizedDataFrame.show()

    spark.stop()
