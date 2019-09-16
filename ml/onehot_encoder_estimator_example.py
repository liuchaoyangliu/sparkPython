
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("OneHotEncoderEstimatorExample")\
        .getOrCreate()

    # Note: categorical features are usually first encoded with StringIndexer
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
    model = encoder.fit(df)
    encoded = model.transform(df)
    encoded.show()

    spark.stop()
