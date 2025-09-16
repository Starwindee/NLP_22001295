from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.representations.count_vectorizer import CountVectorizer

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


if __name__ == "__main__":
    test_count_vectorizer()
    