from src.representations.word_embedder import WordEmbedder

def main():
    we = WordEmbedder("glove-wiki-gigaword-100")
    
    print("Vector for 'king':")
    print(we.get_vector("king"))
    
    print("\nSimilarity between 'king' and 'queen':", we.get_similarity("king", "queen"))
    print("\nSimilarity between 'king' and 'queen':", we.get_similarity("king", "man"))
    
    print("\nMost similar words to 'computer':")
    print(we.most_similarity("computer", top_n=10))
    
    print("\nDocument embedding for sentence:")
    doc_vec = we.embed_document("The cat is sleeping peacefully on the sunny windowsill.")
    print(doc_vec)

if __name__ == "__main__":
    main()
