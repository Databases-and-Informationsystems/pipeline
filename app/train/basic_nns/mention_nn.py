import os
import typing

from app.model.document import Document
from app.model.schema import Schema
from app.model.settings import ModelSize
from app.util.llm_util import get_prediction, extract_json


class MentionBasicNN:
    size: ModelSize

    def __init__(self, size: ModelSize = ModelSize.MEDIUM):
        self.size = size

    def train(self, schema: Schema, documents: typing.List[Document]) -> str:
        print("mention trained")
        return "trained"

    def evaluate(self, schema: Schema, documents: typing.List[Document]) -> str:
        print("mention evaluated")
        return "evaluated"

    def save(self) -> bool:
        print("saved")
        return True
