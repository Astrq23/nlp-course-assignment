from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
import datetime
import os
from pyspark.sql.functions import col

def main():
    print('before')
    spark = SparkSession.builder.appName("lab03").getOrCreate()
    print('after')
    data_path = "c4-train.00000-of-01024-30K.json"
    df = spark.read.json(data_path).limit(100)
    df = df.select("text").na.drop()

    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

    model = pipeline.fit(df)
    result = model.transform(df)

    os.makedirs("results", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    result.select(col("features").cast("string")).write.mode("overwrite").text("results/lab17_pipeline_output.txt")


    with open("log/lab17_log.txt", "w") as f:
        f.write(f"Start time: {datetime.datetime.now()}\n")
        f.write(f"Input: {data_path}\n")
        f.write(f"Output: results/lab17_pipeline_output.txt\n")
        f.write(f"End time: {datetime.datetime.now()}\n")


main()