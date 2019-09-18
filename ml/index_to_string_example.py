# from pyspark.ml.feature import IndexToString, StringIndexer
# from pyspark.sql import SparkSession
#
# if __name__ == "__main__":
#
#     spark = SparkSession.builder.appName("IndexToStringExample").getOrCreate()
#
#     df = spark.createDataFrame(
#         [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
#         ["id", "category"])
#
#     indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
#     model = indexer.fit(df)
#     indexed = model.transform(df)
#
#     print("将字符串列'％s'转换为索引列 '%s'"
#           % (indexer.getInputCol(), indexer.getOutputCol()))
#     indexed.show()
#
#     print("StringIndexer将标签存储在输出列元数据中\n")
#
#     converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
#     converted = converter.transform(indexed)
#
#     print("使用将已转换的索引列'％s'转换回原始字符串列'％s'"
#           "元数据中的标签" % (converter.getInputCol(), converter.getOutputCol()))
#     converted.select("id", "categoryIndex", "originalCategory").show()
#
#     spark.stop()
from pyspark.ml.feature import StringIndexer, StringIndexerModel, IndexToString
from pyspark.sql import SparkSession, DataFrame

if __name__ == "__main__":

    spark = SparkSession.builder.appName("IndexToString").master("local").getOrCreate()

    df = spark.createDataFrame([
        (0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")
    ], ["id", "category"])

    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    model: StringIndexerModel = indexer.fit(df)
    indexed: DataFrame = model.transform(df)

    indexed.show()

    converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
    convered = converter.transform(indexed)
    convered.show()

    spark.stop()

