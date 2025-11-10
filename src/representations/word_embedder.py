import gensim.downloader as api
from src.preprocessing.regex_tokenizer import RegexTokenizer
import numpy as np


class WordEmbedder:
    def __init__(self, model_name: str):
        try:
            self.model = api.load(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

    def get_vector(self, word: str):
        try:
            return self.model[word]
        except KeyError:
            print(f"{word} is not in vocab!")
            return None

    def get_similarity(self, word1: str, word2: str):
        try:
            return self.model.similarity(word1, word2)
        except KeyError as e:
            print(f"One of the words is not in vocab: {e}")
            return None

    def most_similarity(self, word: str, top_n:int = 10):
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            print(f"{word} is not in vocab!")
            return None
        
    def embed_document(self, document: str):
        tokens = RegexTokenizer().tokenize(document)
        vectors = []

        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(self.model.vector_size)

        return np.mean(vectors, axis=0)
    
    