import gensim.downloader as api
from gensim.models import KeyedVectors
import os
import logging


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
            Word2VecModel._model = KeyedVectors.load(
                f"{self._model_directory}/{self._model_name}"
            )
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
