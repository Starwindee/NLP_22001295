from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.core.dataset_loaders import load_raw_dataset_json

def test_count_vectorizer():
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    X = vectorizer.fit_transform(corpus)

    print("Vocabulary:", vectorizer.vocabulary_)

    print("Document-term matrix:")
    for row in X:
        print(row)

    corpus_json = load_raw_dataset_json(r"E:\Năm 4\NLP\level 1\c4-train.00000-of-01024-30K.json")
    X_json = vectorizer.transform(corpus_json[:3])  
    print("Document-term matrix for JSON corpus:")
    for row in X_json:
        print(row)


if __name__ == "__main__":
    test_count_vectorizer()
    