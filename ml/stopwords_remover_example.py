
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SparkSession



# from pyspark.ml.feature import StopWordsRemover
# from pyspark.sql import SparkSession
#
# if __name__ == "__main__":
#     spark = SparkSession \
#         .builder \
#         .appName("StopWordsRemover") \
#         .master("local") \
#         .getOrCreate()
#
#     sentenceData = spark.createDataFrame([
#         (0, ["I", "saw", "the", "red", "balloon"]),
#         (1, ["Mary", "had", "a", "little", "lamb"])
#     ], ["id", "raw"])
#
#     remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
#     remover.transform(sentenceData).show(truncate=False)
#
#     spark.stop()


if __name__ == "__main__":

    spark = SparkSession.builder.appName("stopWordsRemove").master("local").getOrCreate()

    data = spark.createDataFrame([
        (0, ["I", "saw", "the", "red", "balloon"]),
        (1, ["Mary", "had", "a", "little", "lamb"])
    ], ["id", "raw"])

    remover = StopWordsRemover(inputCol="raw", outputCol="filtered")

    remover.transform(data).show(truncate=False)

    spark.stop()

