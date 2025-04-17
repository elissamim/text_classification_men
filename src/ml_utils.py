import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
import string
import spacy

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("french"))

def nltk_text_preprocessing(x: str) -> str:
    """
    Preprocess textual data using tokenization, pucntuation removal and lowering cases.

    Args:
        x (str): Text to preprocess.

    Returns:
        str: Preprocessed text.
    """

    # Text in lowercase
    x = x.lower()

    # Word tokens for French language
    tokens = word_tokenize(x, 
                           language = "french")

    # Stopwords and punctuation removal
    tokens = [
        token for token in tokens
        if token not in stop_words and token not in string.punctuation
    ]

    # French stemming
    stemmer = FrenchStemmer()
    stemmed_tokens = [
        stemmer.stem(token) for token in tokens
    ]

    # French lemmatization

    return " ".join(stemmed_tokens)