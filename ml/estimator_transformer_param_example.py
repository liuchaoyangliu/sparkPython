from __future__ import print_function

import findspark

findspark.init()

from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("EstimatorTransformerParamExample") \
        .master("local") \
        .getOrCreate()

    # 从（标签，功能）元组列表中准备训练数据。
    training = spark.createDataFrame([
        (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

    # 创建LogisticRegression实例。这个实例是一个Estimator。
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    # 打印出参数，文档和任何默认值。
    print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    # 学习LogisticRegression模型。这使用存储在lr中的参数。
    model1 = lr.fit(training)

    # 由于model1是一个模型（即由Estimator生成的变换器），我们可以查看它在fit（）
    # 期间使用的参数。这将打印参数（name：value）对，其中names是thisLogisticRegression实例的唯一ID。
    # print("Model 1 was fit using parameters: ")
    print(model1.extractParamMap())

    # 我们也可以使用Python字典作为paramMap指定参数
    paramMap = {lr.maxIter: 20}
    paramMap[lr.maxIter] = 30  # 指定1个参数，覆盖原始的maxIter。
    paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  # 指定多个参数。

    # 你可以组合paramMaps，它们是python词典。
    paramMap2 = {lr.probabilityCol: "myProbability"}  # 更改输出列名称
    paramMapCombined = paramMap.copy()
    paramMapCombined.update(paramMap2)

    # 现在使用paramMapCombined参数学习一个新模型。
    # paramMapCombined通过lr.set *方法覆盖之前设置的所有参数。
    model2 = lr.fit(training, paramMapCombined)
    print("Model 2 was fit using parameters: ")
    print(model2.extractParamMap())

    # 准备测试数据
    test = spark.createDataFrame([
        (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
        (0.0, Vectors.dense([3.0, 2.0, -0.1])),
        (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

    # 使用Transformer.transform（）方法对测试数据进行预测。
    #  LogisticRegression.transform只会使用“功能”列。
    #  请注意，model2.transform（）输出“myProbability”列而不是通常的列
    #  'probability'列，因为我们先前重命名了lr.probabilityCol参数。
    prediction = model2.transform(test)
    result = prediction.select("features", "label", "myProbability", "prediction") \
        .collect()

    print()
    print()
    for row in result:
        print("features=%s, label=%s -> prob=%s, prediction=%s"
              % (row.features, row.label, row.myProbability, row.prediction))

    spark.stop()
