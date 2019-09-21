from pyspark.ml.feature import RFormula
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RFormula").getOrCreate()

    dataset = spark.createDataFrame([
        (7, "US", 18, 1.0),
        (8, "CA", 12, 0.0),
        (9, "NZ", 15, 0.0)
    ], ["id", "country", "hour", "clicked"])

    formula = RFormula(
        formula="clicked ~ country + hour",
        featuresCol="features",
        labelCol="label")

    output = formula.fit(dataset).transform(dataset)
    output.show()
    # output.select("features", "label").show()

    spark.stop()
