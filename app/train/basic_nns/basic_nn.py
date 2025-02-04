import os
import typing
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod
from enum import Enum

from app.train.basic_nns.basic_nn_utils import get_token_postag_list, get_mention_tag_list, get_relation_tag_list
from app.model.document import Document, Token, Mention, Relation
from app.model.schema import Schema
from app.model.settings import ModelSize
from app.word_embeddings.word2vec import Word2VecModel


class BasicNNType(Enum):
    MENTION_NN = "mentions"
    ENTITY_NN = "entities"
    RELATION_NN = "relations"


class BasicNN(nn.Module, ABC):
    size: ModelSize
    word2vec: Word2VecModel
    _nn_type: BasicNNType
    token_postag_list: list[str]
    mention_tag_list: list[str]
    relation_tag_list: list[str]

    def __init__(
        self,
        nn_type: BasicNNType,
        size: ModelSize = ModelSize.MEDIUM,
        documents: typing.List[Document] = [],
        schema_id: typing.Optional[str] = None,
    ):
        super(BasicNN, self).__init__()
        self._nn_type = nn_type
        self.size = size
        self.word2vec = Word2VecModel()
        self.token_postag_list = get_token_postag_list(
            documents=documents
        )
        self.mention_tag_list = get_mention_tag_list(documents=documents)
        self.relation_tag_list = get_relation_tag_list(
            documents=documents
        )

        if schema_id is not None:
            self._load_from_file(schema_id)
        else:
            self._init_layer()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

    def start_training(self, documents: typing.List[Document]) -> str:
        X, y = self._prepare_train_data(documents=documents)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.005)

        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)

        batch_size = 32
        num_samples = X_train.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        num_epochs = 50
        epoch_loss_list = []
        for epoch in range(num_epochs):
            self.train()

            epoch_loss = 0.0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= num_batches
            print(f"Epoche [{epoch+1}/{num_epochs}] abgeschlossen. Loss: {epoch_loss}")
            epoch_loss_list.append(epoch_loss)

        return epoch_loss_list

    def evaluate(self, schema: Schema, documents: typing.List[Document]) -> str:
        random.shuffle(documents)
        num_splits = 5
        split_size = len(documents) // num_splits
        splits = [
            documents[i * split_size : (i + 1) * split_size] for i in range(num_splits)
        ]

        remainder = len(documents) % num_splits
        for i in range(remainder):
            splits[i].append(documents[num_splits * split_size + i])

        score = []
        for fold in range(num_splits):
            test_set = splits[fold]
            train_set = [
                doc for i, split in enumerate(splits) if i != fold for doc in split
            ]

            print(f"Durchgang {fold + 1} von {num_splits}")

            self._init_layer()
            self.start_training(documents=train_set)

            for test_document in test_set:
                if self._nn_type == BasicNNType.ENTITY_NN:
                    prediction = self.predict(test_document.mentions)
                if self._nn_type == BasicNNType.MENTION_NN:
                    prediction = self.predict(test_document.tokens)
                if self._nn_type == BasicNNType.RELATION_NN:
                    prediction = self.predict(test_document.mentions)

                score.append(
                    self._evaluate_prediction_against_truth(
                        prediction=prediction, truth=test_document
                    )
                )
        print(sum(score) / len(score))
        return sum(score) / len(score)

    def _get_hidden_layer_sizes(self) -> list[int]:
        if self.size == ModelSize.SMALL:
            return [50, 25, 15, 10]
        elif self.size == ModelSize.BIG:
            return [200, 100, 50, 25]
        elif self.size == ModelSize.MEDIUM:
            return [100, 50, 30, 10]
        else:
            raise ValueError(f"ModelSize '{self.size}' is not supported.")

    def save_as_file(self, schema_id):
        directory = f"basic_nn/{self._nn_type.value}/{schema_id}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path_nn = f"{directory}/nn.pth"
        file_path_metadata = f"{directory}/metadata.json"

        metadata = {
            "token_postag_list": self.token_postag_list,
            "mention_tag_list": self.mention_tag_list,
            "relation_tag_list": self.relation_tag_list,
            "layer_sizes": [
                self.fc1.in_features,
                self.fc1.out_features,
                self.fc2.out_features,
                self.fc3.out_features,
                self.fc4.out_features,
                self.fc5.out_features,
            ],
        }

        torch.save(self.state_dict(), file_path_nn)
        with open(file_path_metadata, "w") as metadata_file:
            json.dump(metadata, metadata_file)

        print(f"modell saved in: {file_path_nn}")
        print(f"metadata saved in: {file_path_metadata}")

    def _load_from_file(self, schema_id: str):
        directory = f"basic_nn/{self._nn_type.value}/{schema_id}"
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Pfad nicht gefunden: {directory}")

        file_path_metadata = f"{directory}/metadata.json"
        file_path_model = f"{directory}/nn.pth"

        if not os.path.exists(file_path_model):
            raise FileNotFoundError(f"Model file not found: {file_path_model}")

        if not os.path.exists(file_path_metadata):
            raise FileNotFoundError(f"Metadata file not found: {file_path_metadata}")

        with open(file_path_metadata, "r") as metadata_file:
            metadata = json.load(metadata_file)

        self.token_postag_list = metadata["token_postag_list"]
        self.mention_tag_list = metadata["mention_tag_list"]
        self.relation_tag_list = metadata["relation_tag_list"]

        layer_sizes = metadata["layer_sizes"]
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        self.fc5 = nn.Linear(layer_sizes[4], layer_sizes[5])

        self.load_state_dict(torch.load(file_path_model))
        print(f"Model successfully loaded from {file_path_model}")

    def _get_mention_tag_nn_input_list(self, mention: Mention):
        nn_tag_list = []
        if mention is None:
            return [0] * len(self.mention_tag_list)
        for tag in self.mention_tag_list:
            if mention.tag == tag:
                nn_tag_list.append(1)
            else:
                nn_tag_list.append(0)
        return nn_tag_list

    def _get_relation_tag_nn_input_list(self, relation: Relation):
        nn_tag_list = []
        if relation is None:
            return [0] * len(self.relation_tag_list)
        for tag in self.relation_tag_list:
            if relation.tag == tag:
                nn_tag_list.append(1)
            else:
                nn_tag_list.append(0)
        return nn_tag_list

    def _get_token_postag_nn_input_list(self, token: Token):
        nn_input_list = []
        for postag in self.token_postag_list:
            if token.pos_tag == postag:
                nn_input_list.append(1)
            else:
                nn_input_list.append(0)
        return nn_input_list

    def _get_mention_postag_nn_input_list(self, mention: Mention):
        nn_input_list = None
        for token in mention.tokens:
            postag_list = self._get_token_postag_nn_input_list(token)
            if nn_input_list is None:
                nn_input_list = postag_list
            else:
                nn_input_list = list(np.array(nn_input_list) + np.array(postag_list))
        return nn_input_list

    def _get_mention_text(self, mention: Mention):
        text = ""
        for token in mention.tokens:
            text += f"{token.text} "
        return text

    def _get_string_similarity(self, string0: str, string1: str) -> float:
        set0 = set(string0.lower())
        set1 = set(string1.lower())
        setIntersection = set0.intersection(set1)
        intersection = len(setIntersection)
        setUnion = set0.union(set1)
        union = len(setUnion)
        return intersection / union

    def _init_layer(self):
        input_size, output_size = self._get_input_output_size()
        hidden_sizes = self._get_hidden_layer_sizes()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)

    @abstractmethod
    def _get_input_output_size(self):
        pass

    @abstractmethod
    def _prepare_train_data(self, documents: typing.List[Document]):
        pass

    @abstractmethod
    def predict(self, content):
        pass

    @abstractmethod
    def _evaluate_prediction_against_truth(self, prediction, truth):
        pass
