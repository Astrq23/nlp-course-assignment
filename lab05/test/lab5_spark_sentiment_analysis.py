
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
)
from pyspark.ml.classification import (
    LogisticRegression, NaiveBayes
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def build_spark_session():

    return SparkSession.builder.appName("AdvancedSentimentAnalysis") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate() # 

def load_and_clean_data(spark: SparkSession, data_path: str):
 
    print(f"Đang tải dữ liệu từ {data_path}...")
    df = spark.read.csv(data_path, 
                        header=True, 
                        inferSchema=True, 
                        multiLine=True,    
                        quote='"',         
                        escape='"')      
    df = df.dropna(subset=["text", "label"])

    df_clean = df.withColumn("clean_text", lower(col("text")))
    df_clean = df_clean.withColumn("clean_text", 
        regexp_replace(col("clean_text"), r'http\S+', ''))
    df_clean = df_clean.withColumn("clean_text", 
        regexp_replace(col("clean_text"), r'@[a-zA-Z0-9_]+', ''))
    df_clean = df_clean.withColumn("clean_text", 
        regexp_replace(col("clean_text"), r'[^a-zA-Z\s]', ''))
    
    return df_clean.select(col("clean_text").alias("text"), 
                           col("label").cast("double"))

def get_evaluators():
    """
    Tạo các trình đánh giá (evaluators) cho accuracy và f1.
    """
    accuracy_eval = MulticlassClassificationEvaluator(
        metricName="accuracy", labelCol="label") 
    
    f1_eval = MulticlassClassificationEvaluator(
        metricName="f1", labelCol="label")
    
    return accuracy_eval, f1_eval

def main():
    spark = build_spark_session()
    data_path = "data/twitter_financial_sentiment.csv"
    
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy tệp {data_path}.")
        print("Vui lòng chạy tệp 'prepare_dataset.py' trước.")
        spark.stop()
        return

    df = load_and_clean_data(spark, data_path)
    
    (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)
    print("Dữ liệu đã được chia thành tập train và test.")
    
    acc_eval, f1_eval = get_evaluators()


    print("\nĐang xây dựng Pipeline 1: TF-IDF + Logistic Regression (Baseline)")
    tokenizer_tf = Tokenizer(inputCol="text", outputCol="words")
    stopwords_tf = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.01, featuresCol="features", labelCol="label")
    pipeline_lr = Pipeline(stages=[tokenizer_tf, stopwords_tf, hashingTF, idf, lr])


    print("Đang xây dựng Pipeline 2: TF-IDF + Naive Bayes (Improvement)")
    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
    pipeline_nb = Pipeline(stages=[tokenizer_tf, stopwords_tf, hashingTF, idf, nb])


    print("Đang xây dựng Pipeline 3: Word2Vec + Logistic Regression (Improvement)")
    tokenizer_w2v = Tokenizer(inputCol="text", outputCol="words")
    stopwords_w2v = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="features")
    lr_w2v = LogisticRegression(maxIter=10, regParam=0.01, featuresCol="features", labelCol="label")
    pipeline_w2v = Pipeline(stages=[tokenizer_w2v, stopwords_w2v, word2Vec, lr_w2v])

    pipelines = {
        "Baseline (TF-IDF + LR)": pipeline_lr,
        "Improvement (TF-IDF + NB)": pipeline_nb,
        "Improvement (Word2Vec + LR)": pipeline_w2v,
    }

    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n--- Đang huấn luyện: {name} ---")
        model = pipeline.fit(trainingData) 
        
        print(f"--- Đang đánh giá: {name} ---")
        predictions = model.transform(testData) 
        
        accuracy = acc_eval.evaluate(predictions)
        f1 = f1_eval.evaluate(predictions)
        
        results[name] = {"Accuracy": accuracy, "F1-Score": f1}

    print("\n---  KẾT QUẢ SO SÁNH HIỆU SUẤT ---")
    print("-" * 50)
    print(f"{'Mô hình':<30} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Accuracy']:<10.4f} | {metrics['F1-Score']:<10.4f}")
        
    print("-" * 50)
    
    spark.stop()

if __name__ == "__main__":
    main()