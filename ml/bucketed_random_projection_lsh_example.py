from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("BucketedRandomProjectionLSHExample") \
        .getOrCreate()

    dataA = [(0, Vectors.dense([1.0, 1.0]),),
             (1, Vectors.dense([1.0, -1.0]),),
             (2, Vectors.dense([-1.0, -1.0]),),
             (3, Vectors.dense([-1.0, 1.0]),)]
    dfA = spark.createDataFrame(dataA, ["id", "features"])

    dataB = [(4, Vectors.dense([1.0, 0.0]),),
             (5, Vectors.dense([-1.0, 0.0]),),
             (6, Vectors.dense([0.0, 1.0]),),
             (7, Vectors.dense([0.0, -1.0]),)]
    dfB = spark.createDataFrame(dataB, ["id", "features"])

    key = Vectors.dense([1.0, 0.0])

    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0,
                                      numHashTables=3)
    model = brp.fit(dfA)

    # Feature Transformation
    print("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()

    # Compute the locality sensitive hashes for the input rows, then perform approximate
    # similarity join.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
    print("Approximately joining dfA and dfB on Euclidean distance smaller than 1.5:")
    model.approxSimilarityJoin(dfA, dfB, 1.5, distCol="EuclideanDistance")\
        .select(col("datasetA.id").alias("idA"),
                col("datasetB.id").alias("idB"),
                col("EuclideanDistance")).show()

    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    # neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    print("Approximately searching dfA for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(dfA, key, 2).show()

    spark.stop()
