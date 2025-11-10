import sklearn
from src.core.interfaces import Vectorizer
from typing import List, Dict


class TextClassifier:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self.model = sklearn.linear_model.LogisticRegression() 
    
    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()

    def evaluate(self, y_true : List[int], y_pred : List[int]) -> Dict[str, float]:
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred, average='weighted')
        recall = sklearn.metrics.recall_score(y_true, y_pred, average='weighted')
        f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }