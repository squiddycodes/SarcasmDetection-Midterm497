import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from gensim.models import Word2Vec
nltk.download('punkt')
nltk.download('wordnet')


def preprocess_text(text):
    '''
    Tokenizes and lemmatizes an instance of text for every case.
    '''
    lemmatizer = WordNetLemmatizer()

    if not isinstance(text, str):
        text = str(text)
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return processed_tokens

def load_glove_embeddings(glove_file_path):
    '''
    Builds a dictionary of word embeddings for every word.
    Used to assign embeddings to our corpus
    '''
    embeddings_index = {}
    with open(glove_file_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_combined_embedding(token, glove_embeddings, tfidf_embeddings, word2vec_embeddings, glove_dim, tfidf_dim, word2vec_dim):
    '''
    Concatenates the glove, tfidf, and word2vec vectors into ideally a 300 dimensional space.
    Padding is added later to ensure the model works correctly.
    '''
    glove_vec = glove_embeddings.get(token, np.zeros(glove_dim))
    tfidf_vec = tfidf_embeddings.get(token, np.zeros(tfidf_dim))
    word2vec_vec = word2vec_embeddings.get(token, np.zeros(word2vec_dim))
    return np.concatenate([glove_vec, tfidf_vec, word2vec_vec])

def embed_sentence(tokens, glove_embeddings, tfidf_embeddings, word2vec_embeddings,glove_dim, tfidf_dim, word2vec_dim):
    '''
    Embeds an entire sentence in the 3 respective embeddings styles. Returns an array of these embeddings.
    This is done on a case-by-case basis, and is called for every case in the corpus
    '''
    embeddings = []
    for token in tokens:
        embeddings.append(get_combined_embedding(token, 
                                                 glove_embeddings, 
                                                 tfidf_embeddings,
                                                 word2vec_embeddings, 
                                                 glove_dim, 
                                                 tfidf_dim, 
                                                 word2vec_dim))
    return np.array(embeddings)

def pad_embedding(emb, max_len, combined_dim):
    '''
    Used in order to uniform uniform input size for our deep learning model.
    '''
    L = emb.shape[0]
    if L < max_len:
        padding = np.zeros((max_len - L, combined_dim))
        return np.vstack([emb, padding])
    else:
        return emb[:max_len, :]
    
def get_data():

    #intitial cleaning
    data = pd.read_csv("combined_data.csv", 
            header=None, 
            names=["Text", "Label"], 
            delimiter=',', 
            encoding='utf-8')
    data = data[data['Label'].isin(['0', '1'])]
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    data['Processed'] = data['Text'].apply(preprocess_text)
    data['Text_joined'] = data['Processed'].apply(lambda tokens: " ".join(tokens))

    #creation of glove embeddings
    glove_path = "glove.6B.100d.txt"
    glove_embeddings = load_glove_embeddings(glove_path)
    glove_dim = 100

    #creation of TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Text_joined'])
    vocab = tfidf_vectorizer.vocabulary_

    #Feature hashing to 100 dim space for TF-IDF features
    hash_vectorizer = HashingVectorizer(n_features=100, norm=None, alternate_sign=False)
    tfidf_embeddings = {}
    for word in vocab:
        vector = hash_vectorizer.transform([word]).toarray()[0]
        tfidf_embeddings[word] = vector
    tfidf_dim = 100

    #word2vec feature construction
    word2vec_model = Word2Vec(sentences=data['Processed'], vector_size=100, window=5, min_count=1, workers=4, seed=42)
    word2vec_dim = 100
    word2vec_embeddings = {word: word2vec_model.wv[word] for word in word2vec_model.wv.index_to_key}

    #combining all 3 features into 1 vector
    combined_dim = glove_dim + tfidf_dim + word2vec_dim

    #creates a new column embedded w/ the combined features
    data['Embedded'] = data['Processed'].apply(
        lambda tokens: embed_sentence(tokens, 
                                      glove_embeddings, 
                                      tfidf_embeddings, 
                                      word2vec_embeddings,
                                      glove_dim, 
                                      tfidf_dim, 
                                      word2vec_dim))

    #perform embedding padding for model readiness.
    #Lambdas used to convert the entire corpus and create a new variable of feature vector
    max_len = data['Embedded'].apply(lambda emb: emb.shape[0]).max()
    data['Padded'] = data['Embedded'].apply(lambda emb: pad_embedding(emb, max_len, combined_dim))
    X = np.stack(data['Padded'].values)
    y = data['Label'].astype(int).values

    #train-test split and return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, max_len, combined_dim



