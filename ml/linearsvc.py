from pyspark.ml.classification import LinearSVC
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("linearSVC Example") \
        .getOrCreate()

    # Load training data
    training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    lsvc = LinearSVC(maxIter=10, regParam=0.1)

    # Fit the model
    lsvcModel = lsvc.fit(training)

    # Print the coefficients and intercept for linear SVC
    print("Coefficients: " + str(lsvcModel.coefficients))
    print("Intercept: " + str(lsvcModel.intercept))

    spark.stop()
