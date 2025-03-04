import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import os
import typing

from app.util.logger import logger


class Word2VecModel:
    _model = None
    _model_directory = "word_embedding_models"
    _model_name = "word2vec"
    vector_size = 300

    def __init__(self):
        if not Word2VecModel._model:
            self._load_model()

    def _load_model(self):
        if os.path.exists(f"{self._model_directory}/{self._model_name}"):
            logger.info("load word2vec model...")
            Word2VecModel._model = KeyedVectors.load(
                f"{self._model_directory}/{self._model_name}"
            )
            logger.info("successfully loaded word2vec model")
        else:
            model = api.load("word2vec-google-news-300")
            if os.path.exists(self._model_directory) == False:
                os.mkdir(self._model_directory)
            model.save(f"{self._model_directory}/{self._model_name}")
            Word2VecModel._model = model

    def get_vector(self, word: str):
        if Word2VecModel._model is None:
            raise ValueError("failed to word2vec model")
        try:
            return Word2VecModel._model[word]
        except KeyError:
            return None

    def get_vector_for_multiple_words(self, words: typing.List[str]):
        word_vectors = [
            self.get_vector(word) for word in words if self.get_vector(word) is not None
        ]

        if not word_vectors:
            return None

        return np.mean(word_vectors, axis=0)

    def get_multiple_vector_for_multiple_words(
        self, words: typing.List[str], max_length
    ):
        word_vectors = [
            self.get_vector(word) for word in words if self.get_vector(word) is not None
        ]

        if not word_vectors:
            for i in range(max_length):
                word_vectors.append(np.zeros(max_length * self.vector_size))
                return word_vectors

        if len(word_vectors) > max_length:
            word_vectors = word_vectors[:max_length]

        if len(word_vectors) < max_length:
            original_length = len(word_vectors)
            for i in range(max_length - original_length):
                word_vectors.append(np.zeros(self.vector_size))

        return word_vectors
