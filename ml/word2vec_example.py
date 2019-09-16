
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec

if __name__ == "__main__":
    spark = SparkSession.builder.appName("word2Vec").master("local").getOrCreate()

    documentDF = spark.createDataFrame([
        ("Hi i heard about spark".split(" "),),
        ("I wish Java could use case classes".split(" "),),
        ("Logistic regression models are neat".split(" "),)
    ], ["text"])

    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)

    result = model.transform(documentDF)

    for row in result.collect():
        text, result = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(result)))

    spark.stop()
