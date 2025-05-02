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
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
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
class NMFTopicModeling:
    raw_texts: List[str]
    num_topics: int = 3

    def nmf_preprocessing(self, x: str) -> str:
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

    def nmf_model(self):
        """

        """

        # Preprocess data
        preprocessed_docs = [self.nmf_preprocessing(doc) for doc in self.raw_texts]

        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
        tfidf = vectorizer.fit_transform(preprocessed_docs)
        nmf = NMF(n_components=self.num_topics, random_state=42)
        W =
        H = 
