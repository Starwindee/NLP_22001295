import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split

def main():
    spark = SparkSession.builder.appName("app-nlp-demo").getOrCreate()
    print("Spark session started successfully.")

    data_path = r"E:\Nam4\NLP\level 1\c4-train.00000-of-01024-30K.json"

    print(f"Loading dataset from: {data_path}")
    df = spark.read.json(data_path)
    print(f"Loaded {df.count()} rows of text data.")

    df = df.select("text")
    
    print("Cleaning and tokenizing text...")

    df = df.select(lower(col("text")).alias("text"))
    df = df.select(regexp_replace(col("text"), r"[^a-zA-Z0-9\s]", "").alias("text"))
    df = df.select(split(col("text"), r"\s+").alias("words"))

    print("Training Word2Vec model (100 dimensions)...")
    word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="words", outputCol="vectors")

    model = word2Vec.fit(df)
    print("Word2Vec model trained successfully!")

    print("\nFinding synonyms for 'computer':")
    try:
        synonyms = model.findSynonyms("computer", 5)
        synonyms.show(truncate=False)
    except Exception as e:
        print(f"Could not find synonyms for 'computer': {e}")
    
    spark.stop()

if __name__ == "__main__":
    main()