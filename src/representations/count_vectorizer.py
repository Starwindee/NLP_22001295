from typing import List, Dict
from src.core.interfaces import Vectorizer, Tokenizer


class CountVectorizer(Vectorizer):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}

    def fit(self, corpus: List[str]) -> None:
 
        vocab_set = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            vocab_set.update(tokens)

        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(vocab_set))}

    def transform(self, documents: List[str]) -> List[List[int]]:
   
        vectors = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            vector = [0] * len(self.vocabulary_)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1
            vectors.append(vector)
        return vectors

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Thực hiện fit và transform trên cùng corpus.
        """
        self.fit(corpus)
        return self.transform(corpus)
