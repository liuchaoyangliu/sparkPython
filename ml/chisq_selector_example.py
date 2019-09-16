import findspark

findspark.init()

from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ChiSqSelectorExample")\
        .getOrCreate()

    df = spark.createDataFrame([
        (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
        (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
        (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])

    selector = ChiSqSelector(numTopFeatures=1, featuresCol="features",
                             outputCol="selectedFeatures", labelCol="clicked")

    result = selector.fit(df).transform(df)

    print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
    result.show()

    spark.stop()
