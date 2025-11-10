from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml import Pipeline
import pandas as pd
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec, CountVectorizer
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=False)
    
    # Drop missing labels or text
    df = df.dropna(subset=["text", "label"])
    
    # Chỉ giữ các giá trị label hợp lệ (0,1,2)
    df = df.filter(col("label").rlike("^[0-2]$"))
    
    # Ép kiểu label sang integer
    df = df.withColumn("label", col("label").cast("int"))

    # Text cleaning
    df = df.withColumn("text", lower(col("text")))
    df = df.withColumn("text", regexp_replace(col("text"), r"http\S+|www\S+", ""))
    df = df.withColumn("text", regexp_replace(col("text"), r"<.*?>", ""))
    df = df.withColumn("text", regexp_replace(col("text"), r"[^a-zA-Z\s]", ""))
    df = df.withColumn("text", regexp_replace(col("text"), r"\s+", " "))
    
    return df



def evaluate_model(name, pipeline, train_df, valid_df):
    model = pipeline.fit(train_df)
    predictions = model.transform(valid_df)

    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")

    acc = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    return name, acc, f1


def main():
    spark = SparkSession.builder.appName("Task4_Model_Improvement").getOrCreate()
    print("Spark session started successfully.")

    train_path = r"E:\Nam4\NLP\level 1\NLP_22001295_HUS\data/twitter-financial-news-sentiment/sent_train.csv"
    valid_path = r"E:\Nam4\NLP\level 1\NLP_22001295_HUS\data/twitter-financial-news-sentiment/sent_valid.csv"

    train_df = load_data(spark, train_path)
    valid_df = load_data(spark, valid_path)
    print(f"Train samples: {train_df.count()}, Validation samples: {valid_df.count()}")

    # Common preprocessing
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")

    results = []

    # 1) TF-IDF + Logistic Regression
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(maxIter=20, regParam=0.001, featuresCol="features", labelCol="label", family="multinomial")
    pipeline_tfidf_lr = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    results.append(evaluate_model("TF-IDF + Logistic Regression", pipeline_tfidf_lr, train_df, valid_df))

    # 2) CountVectorizer + Naive Bayes
    count_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features", minDF=2, maxDF=0.8)
    nb = NaiveBayes(featuresCol="features", labelCol="label")
    pipeline_cv_nb = Pipeline(stages=[tokenizer, remover, count_vectorizer, nb])
    results.append(evaluate_model("CountVectorizer + Naive Bayes", pipeline_cv_nb, train_df, valid_df))

    # 3) Word2Vec + Logistic Regression
    word2vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered", outputCol="features")
    lr_word2vec = LogisticRegression(maxIter=20, regParam=0.001, featuresCol="features", labelCol="label", family="multinomial")
    pipeline_word2vec_lr = Pipeline(stages=[tokenizer, remover, word2vec, lr_word2vec])
    results.append(evaluate_model("Word2Vec + Logistic Regression", pipeline_word2vec_lr, train_df, valid_df))

    # 4) Word2Vec + Random Forest
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10)
    pipeline_word2vec_gbt = Pipeline(stages=[tokenizer, remover, word2vec, rf])
    results.append(evaluate_model("Word2Vec + Random Forest", pipeline_word2vec_gbt, train_df, valid_df))

    # Summary
    print("\nSummary of Model Comparison")
    for name, acc, f1 in results:
        print(f"{name:<35} Accuracy: {acc:.4f} | F1: {f1:.4f}")

    spark.stop()
    print("\nSpark session stopped.")

if __name__ == "__main__":
    main()
