import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
import string
import spacy
import spacy.cli
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import List

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
spacy.cli.download("fr_core_news_sm")
nlp = spacy.load("fr_core_news_sm")
stop_words = set(stopwords.words("french"))


@dataclass
class LDATopicModeling:
    raw_texts: List[str]
    num_topics_inf: int = 1
    num_topics_sup: int = 10
    num_topics_step: int = 1
    lemmatization: bool = True
    num_words_per_topics_word_cloud: int = 30

    def lda_preprocessing(self, x: str) -> List[str]:
        """
        Preprocess textual data using tokenization, pucntuation removal and lowering cases.

        Args:
            x (str): Text to preprocess.

        Returns:
            List[str]: List of tokens.
        """

        # Text in lowercase
        x = x.lower()
        # Word tokens for French language
        tokens = word_tokenize(x, language="french")
        # Stopwords and punctuation removal
        tokens = [
            token
            for token in tokens
            if token not in stop_words and token not in string.punctuation + "..."
        ]
        # French stemming
        if not self.lemmatization:
            stemmer = FrenchStemmer()
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            return stemmed_tokens

        # French lemmatization
        doc = nlp(" ".join(tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]
        return lemmatized_tokens

    def find_best_lda_model(self) -> gensim.models.ldamodel.LdaModel:
        """
        Finds the best LDA model in terms of Coherence Value for a
        given list of strings (documents) and for different values of
        number of topics.

        Returns:
            gensim.models.ldamodel.LdaModel: Best LDA model in terms of Coherence Value.
        """

        # Preprocess documents to get a list of lists of tokens
        preprocessed_docs = [self.lda_preprocessing(doc) for doc in self.raw_texts]

        # Create a dictionary
        dictionary = corpora.Dictionary(preprocessed_docs)

        # Bag of Words feature extraction from the dictionary
        corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

        # Find LDA model with highest Coherence Value
        coherence_values = []
        model_list = []

        for num_topics in range(
            self.num_topics_inf, self.num_topics_sup, self.num_topics_step
        ):
            model = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha="auto",
                per_word_topics=True,
            )
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model,
                texts=preprocessed_docs,
                dictionary=dictionary,
                coherence="c_v",
            )
            coherence_values.append(coherencemodel.get_coherence())

        return model_list[np.argmax(coherence_values)]

    def plot_word_clouds(self, lda_model: gensim.models.ldamodel.LdaModel) -> None:
        """
        Plot a Word Cloud for each topic of a given LDA model.

        Args:
            lda_model (gensim.models.ldamodel.LdaModel): LDA model.

        Returns:
            None: Word Clouds for each topic of the LDA model.
        """

        for topic in range(lda_model.num_topics):
            plt.figure()
            plt.title(f"Topic {topic}")
            word_freqs = dict(
                lda_model.show_topic(topic, self.num_words_per_topics_word_cloud)
            )
            word_cloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate_from_frequencies(word_freqs)
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()

@dataclass
class DeterministicTopicModeling:
    lemmatization: bool = True
    
    def preprocessing(self, x: str) -> str:
        """
        Preprocess textual data using tokenization, pucntuation removal and lowering cases.
    
        Args:
            x (str): Text to preprocess.
    
        Returns:
            str: Preprocessed string.
        """
    
        # Text in lowercase
        x = x.lower()
        # Word tokens for French language
        tokens = word_tokenize(x, language="french")
        # Stopwords and punctuation removal
        tokens = [
            token
            for token in tokens
            if token not in stop_words and token not in string.punctuation + "..."
        ]
        # French stemming
        if not self.lemmatization:
            stemmer = FrenchStemmer()
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            return " ".join(stemmed_tokens)
    
        # French lemmatization
        doc = nlp(" ".join(tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]
        return " ".join(lemmatized_tokens)
    
@dataclass
class NMFTopicModeling(DeterministicTopicModeling):
    raw_texts: List[str]
    num_topics: int = 3
    num_top_words: int = 10

    preprocessed_docs: List[str] = field(init=False)
    vectorizer: TfidfVectorizer = field(init=False)
    tfidf: any = field(init=False)
    nmf: NMF = field(init=False)
    W: np.ndarray = field(init=False)
    H: np.ndarray = field(init=False)
    feature_names: List[str] = field(init=False)

    def __post_init__(self):
        self.nmf_model()

    def nmf_model(self):
        """
        Fit the NMF model.
        """

        # Preprocess data
        self.preprocessed_docs = [self.preprocessing(doc) for doc in self.raw_texts]

        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
        self.tfidf = self.vectorizer.fit_transform(self.preprocessed_docs)

        # Get the decomposition
        self.nmf = NMF(n_components=self.num_topics, random_state=42)
        self.W = self.nmf.fit_transform(self.tfidf)
        self.H = self.nmf.components_

        self.feature_names = self.vectorizer.get_feature_names_out()

    def print_topics(self):
        """
        Print the top words for each topic.
        """

        for topic_idx, topic in enumerate(self.H):
            print(f"Topic {topic_idx}:")
            top_words = [
                self.feature_names[i]
                for i in topic.argsort()[: -self.num_top_words - 1 : -1]
            ]
            print(" ".join(top_words), "\n")

    def plot_word_clouds(self):
        """
        Plot a word cloud for each topic.
        """
        for topic_idx, topic in enumerate(self.H):
            top_indices = topic.argsort()[: -self.num_top_words - 1 : -1]
            freqs = {self.feature_names[i]: topic[i] for i in top_indices}
            wordcloud = WordCloud(
                background_color="white", width=800, height=400
            ).generate_from_frequencies(freqs)

            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Topic {topic_idx}")
            plt.show()

@dataclass
class LSATopicModeling(DeterministicTopicModeling):
    raw_texts: List[str]
    num_topics: int = 3
    num_top_words: int = 10

    preprocessed_docs: List[str] = field(init=False)
    vectorizer: TfidfVectorizer = field(init=False)
    tfidf: any = field(init=False)
    lsa: TruncatedSVD = field(init=False)
    lsa_topic_matrix: np.ndarray = field(init=False)
    feature_names: List[str] = field(init=False)

    def __post_init__(self):
        self.lsa_model()

    def lsa_model(self):
        """
        Fit the LSA model.
        """

        self.preprocessed_docs = [self.preprocessing(doc) for doc in self.raw_texts]

        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
        self.tfidf = self.vectorizer.fit_transform(self.preprocessed_docs)

        self.lsa = TruncatedSVD(n_components = self.num_topics,
                               random_state = 42)
        self.lsa_topic_matrix = self.lsa.fit_transform(self.tfidf)

        self.feature_names = self.vectorizer.get_feature_names_out()

    def print_topics(self):
        """
        Print the top words for each topic.
        """

        for i, comp in enumerate(self.lsa.components_):
            terms_in_topic = [self.feature_names[j] for j in comp.argsort()[:-(self.num_top_words+1):-1]]
            print(f"Topic {i}: {' '.join(terms_in_topic)}")

    def plot_word_clouds(self):
        """
        Plot a word cloud for each topic.
        """
        for i, comp in enumerate(self.lsa.components_):
            top_indices = comp.argsort()[:-(self.num_top_words + 1):-1]
            freqs = {self.feature_names[j]: comp[j] for j in top_indices}
            wordcloud = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(freqs)

            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Topic {i}")
            plt.show()
        