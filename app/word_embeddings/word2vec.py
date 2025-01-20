import gensim.downloader as api
from gensim.models import KeyedVectors
import os
import logging


class Word2VecModel:
    _model = None
    _model_path = "local_word2vec"
    vector_size = 300

    def __init__(self):
        if not Word2VecModel._model:
            self._load_model()

    def _load_model(self):
        if os.path.exists(Word2VecModel._model_path):
            Word2VecModel._model = KeyedVectors.load(Word2VecModel._model_path)
        else:
            model = api.load("word2vec-google-news-300")
            model.save(Word2VecModel._model_path)
            Word2VecModel._model = model

    def get_vector(self, word: str):
        if Word2VecModel._model is None:
            raise ValueError("failed to word2vec model")

        try:
            return Word2VecModel._model[word]
        except KeyError:
            print(f"'{word}' not found")
            return None
