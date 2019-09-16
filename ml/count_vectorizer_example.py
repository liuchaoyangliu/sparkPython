from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.sql import SparkSession


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("CountVectorizerExample") \
        .master("local") \
        .getOrCreate()

    # 输入数据：每行都是一包带有ID的单词。
    df = spark.createDataFrame([
        (0, "a b c".split(" ")),
        (1, "a b b c a".split(" "))
    ], ["id", "words"])

    # 适合语料库中的CountVectorizerModel。
    cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)

    model = cv.fit(df)

    result = model.transform(df)
    result.show(truncate=False)

    spark.stop()
