from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("IsotonicRegressionExample") \
        .getOrCreate()

    # Loads data.
    dataset = spark.read.format("libsvm") \
        .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")

    # Trains an isotonic regression model.
    model = IsotonicRegression().fit(dataset)
    print("Boundaries in increasing order: %s\n" % str(model.boundaries))
    print("Predictions associated with the boundaries: %s\n" % str(model.predictions))

    # Makes predictions.
    model.transform(dataset).show()

    spark.stop()
