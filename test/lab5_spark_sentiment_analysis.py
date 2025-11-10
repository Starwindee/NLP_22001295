import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def load_data(spark, data_path):
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.dropna(subset=["sentiment"])  
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    return df


def build_pipeline():
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])
    return pipeline


def evaluate_model(predictions):
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")


def main():
    spark = SparkSession.builder.appName("Spark-Sentiment-Analysis").getOrCreate()
    print("Spark session started successfully.")

    data_path = r"E:\Nam4\NLP\level 1\NLP_22001295_HUS\data\sentiments.csv"
    df = load_data(spark, data_path)

    # Chia train/test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training samples: {train_df.count()}, Testing samples: {test_df.count()}")

    pipeline = build_pipeline()
    model = pipeline.fit(train_df)
    print("Model training completed.")

    predictions = model.transform(test_df)
    print("Predictions on test data completed.")

    print("\nEvaluation Results:")
    evaluate_model(predictions)

    spark.stop()
    print("Spark session stopped.")


if __name__ == "__main__":
    main()
