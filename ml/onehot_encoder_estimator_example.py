# from pyspark.ml.feature import OneHotEncoderEstimator
# from pyspark.sql import SparkSession
#
# if __name__ == "__main__":
#
#     spark = SparkSession.builder.appName("OneHotEncoderEstimator").master("local").getOrCreate()
#
#     # 注意：分类功能通常首先使用StringIndexer进行编码
#     df = spark.createDataFrame([
#         (0.0, 1.0),
#         (1.0, 0.0),
#         (2.0, 1.0),
#         (0.0, 2.0),
#         (0.0, 1.0),
#         (2.0, 0.0)
#     ], ["categoryIndex1", "categoryIndex2"])
#
#     encoder = OneHotEncoderEstimator(inputCols=["categoryIndex1", "categoryIndex2"],
#                                      outputCols=["categoryVec1", "categoryVec2"])
#     model = encoder.fit(df)
#     encoded = model.transform(df)
#     encoded.show()
#
#     spark.stop()

from pyspark.ml.feature import OneHotEncoderEstimator, OneHotEncoderModel
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession.builder.appName("ontHotEncodeEstimator").getOrCreate()

    df = spark.createDataFrame([
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (2.0, 0.0)
    ], ["categoryIndex1", "categoryIndex2"])

    encoder = OneHotEncoderEstimator(inputCols=["categoryIndex1", "categoryIndex2"],
                                     outputCols=["categoryVec1", "categoryVec2"])

    model: OneHotEncoderModel = encoder.fit(df)

    model.transform(df).show()

    spark.stop()
