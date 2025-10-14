from gensim.models import Word2Vec
import os

def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                yield tokens

def main():
    data_path = r"E:\Nam4\NLP\level 1\UD_English-EWT\en_ewt-ud-train.txt"
    output_path = "results/word2vec_ewt.model"

    if not os.path.exists("results"):
        os.makedirs("results")
    
    print("Reading sentences from data...")
    sentences = list(read_sentences(data_path))

    print("Training Word2Vec model...")
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
    model.save(output_path)
    print("Total words in vocabulary:", len(model.wv))
    print(f"Model saved to {output_path}")

    print("Example usage of the trained model:")
    w = "dog"
    try:
        similar_words = model.wv.most_similar(w, topn=5)
        print(f"\nTop 5 words similar to '{w}':")
        for word, score in similar_words:
            print(f"  {word}: {score:.4f}")
    except KeyError:
        print(f"Word '{w}' not in vocabulary.")


    try:
        analogy = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
        print(f"\nAnalogy example (king - man + woman): {analogy[0]}")
    except KeyError:
        print("Some analogy words not in vocabulary.")

if __name__ == "__main__":
    main()
