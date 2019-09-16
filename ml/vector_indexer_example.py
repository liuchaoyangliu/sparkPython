import findspark

findspark.init()

from __future__ import print_function

from pyspark.ml.feature import VectorIndexer
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("VectorIndexerExample")\
        .getOrCreate()

    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
    indexerModel = indexer.fit(data)

    categoricalFeatures = indexerModel.categoryMaps
    print("Chose %d categorical features: %s" %
          (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))

    # Create new column "indexed" with categorical values transformed to indices
    indexedData = indexerModel.transform(data)
    indexedData.show()

    spark.stop()
