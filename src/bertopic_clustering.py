from dataclasses import dataclass, field
import re
from typing import List
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = list(set(stopwords.words("french")))

@dataclass
class BERTopicClustering:
    raw_texts: List[str]
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    language: str = "french"

    embedding_model: SentenceTransformer = field(init=False)
    vectorizer_model: CountVectorizer = field(init=False)
    umap_model: UMAP = field(init=False)
    hdbscan_model : hdbscan.HDBSCAN = field(init=False)

    def __post_init__(self):
        self.embedding_model = self.load_embedding_model()
        self.vectorizer_model = self.c_tf_idf_model()
        self.umap_model = self.dimension_reduction_model()
        self.hdbscan_model = self.clustering_model()

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

    def dimension_reduction_model(self) -> UMAP:
        """
        Returns a UMAP model for dimension reduction.

        Returns:
            UMAP: a UMAP model.
        """

        umap_model = UMAP(n_neighbors=15,
                         n_components=10,
                         min_dist=0.0,
                         metric="cosine")

        return umap_model

    def clustering_model(self) -> hdbscan.HDBSCAN:
        """
        Returns a HDBSCAN model for clustering.

        Returns:
            hdbscan.HDBSCAN: HDBSCAN clustering model.
        """

        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5,
                                        min_samples=1,
                                        metric="euclidean",
                                        cluster_selection_method="eom",
                                        prediction_data=True)
        
        return hdbscan_model

    def bertopic_model(self):
        """
        Fits a BERTopic model given an embedding model,
        a vectorizer model, a UMAP model, a HDBSCAN model 
        and textual data, for a given language.
        """

        self.bertopic = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            language=self.language
        )

        preprocessed_texts = [self.light_preprocessing(text) for text in self.raw_texts]
        
        return self.bertopic.fit_transform(preprocessed_texts)