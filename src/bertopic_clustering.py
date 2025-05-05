from dataclasses import dataclass
import re
from typing import List
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import nltk
from nltk.corpus import stopwords

nltk.downlaod("stopwords")
stop_words = list(set(stopwords.words("french")))

@dataclass
class BERTopicClustering:
    raw_texts: List[str]
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    embedding_model: SentenceTransformer = field(init=False)
    vectorizer_model: CountVectorizer = field(init=False)

    def __post_init__(self):
        self.embedding_model = self.load_embedding_model()
        self.vectorizer_model = self.c_tf_idf_model()

    def light_preprocessing(self, x:str) -> str:
        """
        Applies a light preprocessing protocol to textual data : removes 
        excessive whitespaces and URLS.

        Args:
            x (str): The text to change.

        Returns:
            str: Preprocessed text.
        """

        clean_text = re.sub(r"\s+", " ", x)
        clean_text = re.sub(r"http\S+", "", x)
        return clean_text.strip()

    def load_embedding_model(self) -> SentenceTransformer:
        """
        Returns a Sentence Transformer for a given embedding model name.

        Returns:
            SentenceTransformer: A Sentence Transformer model for emebddings.
        """

        embedding_model = SentenceTransformer(self.embedding_model_name)

        return embedding_model

    def c_tf_idf_model(self) -> CountVectorizer:
        """
        Returns a c-TF-IDF model.

        Returns:
            CountVectorizer: c-TF-IDF model.
        """

        vectorizer_model = CountVectorizer(ngram_range=(1,2),
                                          stop_words=stop_words)

        return vectorizer_model
        