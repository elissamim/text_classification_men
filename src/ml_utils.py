import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
import string
import spacy
import spacy.cli
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

spacy.cli.download("fr_core_news_sm")
nlp = spacy.load("fr_core_news_sm")

stop_words = set(stopwords.words("french"))

def nltk_text_preprocessing(x: str, lemmatization: bool = False) -> str:
    """
    Preprocess textual data using tokenization, pucntuation removal and lowering cases.

    Args:
        x (str): Text to preprocess.
        lemmatization (bool, Optional): Apply lemmatization (True) or not (False). Defaults to False.

    Returns:
        str: Preprocessed text.
    """

    # Text in lowercase
    x = x.lower()

    # Word tokens for French language
    tokens = word_tokenize(x, language="french")

    # Stopwords and punctuation removal
    tokens = [
        token
        for token in tokens
        if token not in stop_words and token not in string.punctuation
    ]

    # French stemming
    if not lemmatization:
        stemmer = FrenchStemmer()

        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        return " ".join(stemmed_tokens)

    # French lemmatization
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)


class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.vector_size = model.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [
                np.mean(
                    [
                        self.model.wv[word]
                        for word in doc.split()
                        if word in self.model.wv
                    ]
                    or [np.zeros(self.vector_size)],
                    axis=0,
                )
                for doc in X
            ]
        )
