from sklearn.model_selection import train_test_split
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.models.text_classifier import TextClassifier


def load_data():
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0]
    return texts, labels


def train_and_evaluate(texts, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)
    classifier = TextClassifier(vectorizer)

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    evaluation = classifier.evaluate(y_test, predictions)

    return predictions, evaluation


def main():
    texts, labels = load_data()
    predictions, evaluation = train_and_evaluate(texts, labels)

    print("Predictions:", predictions)

    print("\nEvaluation Metrics:")
    for metric, value in evaluation.items():
        print(f"{metric}: {value:.4f}")



if __name__ == "__main__":
    main()
