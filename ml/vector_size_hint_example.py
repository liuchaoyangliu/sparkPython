
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import (VectorSizeHint, VectorAssembler)
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("VectorSizeHintExample")\
        .getOrCreate()

    dataset = spark.createDataFrame(
        [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0),
         (0, 18, 1.0, Vectors.dense([0.0, 10.0]), 0.0)],
        ["id", "hour", "mobile", "userFeatures", "clicked"])

    sizeHint = VectorSizeHint(
        inputCol="userFeatures",
        handleInvalid="skip",
        size=3)

    datasetWithSize = sizeHint.transform(dataset)
    print("Rows where 'userFeatures' is not the right size are filtered out")
    datasetWithSize.show(truncate=False)

    assembler = VectorAssembler(
        inputCols=["hour", "mobile", "userFeatures"],
        outputCol="features")

    # This dataframe can be used by downstream transformers as before
    output = assembler.transform(datasetWithSize)
    print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(truncate=False)

    spark.stop()
