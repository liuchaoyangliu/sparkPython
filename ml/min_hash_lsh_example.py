
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("MinHashLSHExample") \
        .getOrCreate()

    dataA = [(0, Vectors.sparse(6, [0, 1, 2], [1.0, 1.0, 1.0]),),
             (1, Vectors.sparse(6, [2, 3, 4], [1.0, 1.0, 1.0]),),
             (2, Vectors.sparse(6, [0, 2, 4], [1.0, 1.0, 1.0]),)]
    dfA = spark.createDataFrame(dataA, ["id", "features"])

    dataB = [(3, Vectors.sparse(6, [1, 3, 5], [1.0, 1.0, 1.0]),),
             (4, Vectors.sparse(6, [2, 3, 5], [1.0, 1.0, 1.0]),),
             (5, Vectors.sparse(6, [1, 2, 4], [1.0, 1.0, 1.0]),)]
    dfB = spark.createDataFrame(dataB, ["id", "features"])

    key = Vectors.sparse(6, [1, 3], [1.0, 1.0])

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(dfA)

    # Feature Transformation
    print("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()

    # Compute the locality sensitive hashes for the input rows, then perform approximate
    # similarity join.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
    print("Approximately joining dfA and dfB on distance smaller than 0.6:")
    model.approxSimilarityJoin(dfA, dfB, 0.6, distCol="JaccardDistance")\
        .select(col("datasetA.id").alias("idA"),
                col("datasetB.id").alias("idB"),
                col("JaccardDistance")).show()

    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    # neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    # It may return less than 2 rows when not enough approximate near-neighbor candidates are
    # found.
    print("Approximately searching dfA for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(dfA, key, 2).show()

    spark.stop()
