import os
import typing
import json

import torch
import torch.nn as nn
import torch.optim as optim

from app.model.document import Document, Token, Mention, CMention
from app.model.schema import Schema
from app.model.settings import ModelSize
from app.util.llm_util import get_prediction, extract_json
from app.word_embeddings.word2vec import Word2VecModel


class MentionBasicNN(nn.Module):
    size: ModelSize
    word2vec: Word2VecModel
    token_postag_list: list[str]
    mention_tag_list: list[str]

    def __init__(
        self,
        size: ModelSize = ModelSize.MEDIUM,
        documents: typing.List[Document] = [],
        schema_id: typing.Optional[str] = None,
    ):
        self.size = size
        self.token_postag_list = self.get_token_postag_list(documents=documents)
        self.mention_tag_list = self.get_mention_tag_list(documents=documents)
        self.word2vec = Word2VecModel()

        super(MentionBasicNN, self).__init__()
        if schema_id is not None:
            self.load_from_file(schema_id)
        else:
            self.fc1 = nn.Linear(
                4 + 2 * len(self.token_postag_list) + 2 * self.word2vec.vector_size, 100
            )
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, 30)
            self.fc4 = nn.Linear(30, 10)
            self.fc5 = nn.Linear(10, 3 + 2 * len(self.mention_tag_list))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

    def start_training(self, schema: Schema, documents: typing.List[Document]) -> str:
        X, y = self.prepare_train_data(documents=documents)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.005)

        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)

        batch_size = 32
        num_samples = X_train.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        num_epochs = 50
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

        return

    def evaluate(self, schema: Schema, documents: typing.List[Document]) -> str:
        print("mention evaluated")
        # TODO
        return

    def get_mention_by_token(self, document: Document, token_index: int):
        for mention in document.mentions:
            for token in mention.tokens:
                if token.id == token_index:
                    return mention
        return

    def save_as_file(self, schema_id):
        directory = f"basic_nn/mentions/{schema_id}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path_nn = f"{directory}/nn.pth"
        file_path_metadata = f"{directory}/metadata.json"

        metadata = {
            "token_postag_list": self.token_postag_list,
            "mention_tag_list": self.mention_tag_list,
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

    def load_from_file(self, schema_id: str):
        directory = f"basic_nn/mentions/{schema_id}"
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

        layer_sizes = metadata["layer_sizes"]
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        self.fc5 = nn.Linear(layer_sizes[4], layer_sizes[5])

        self.load_state_dict(torch.load(file_path_model))
        print(f"Model successfully loaded from {file_path_model}")

    def get_single_input(self, token0: Token, token1: Token):
        single_X_input = []

        single_X_input.append(token0.document_index)
        single_X_input.append(token1.document_index)

        single_X_input.append(token0.sentence_index)
        single_X_input.append(token1.sentence_index)

        single_X_input += self.get_token_postag_nn_input_list(token0)
        single_X_input += self.get_token_postag_nn_input_list(token1)

        wordvec0 = self.word2vec.get_vector(token0.text)
        wordvec1 = self.word2vec.get_vector(token1.text)

        if wordvec0 is None:
            for i in range(self.word2vec.vector_size):
                single_X_input.append(0)
        else:
            for i in range(self.word2vec.vector_size):
                single_X_input.append(wordvec0[i])

        if wordvec1 is None:
            for i in range(self.word2vec.vector_size):
                single_X_input.append(0)
        else:
            for i in range(self.word2vec.vector_size):
                single_X_input.append(wordvec1[i])

        return single_X_input

    def get_single_output(self, mention0: Mention, mention1: Mention):
        single_y_output = []

        if mention0:
            single_y_output.append(1)
        else:
            single_y_output.append(0)
        if mention1:
            single_y_output.append(1)
        else:
            single_y_output.append(0)

        if mention0 and mention1 and mention0.id == mention1.id:
            single_y_output.append(1)
        else:
            single_y_output.append(0)

        single_y_output += self.get_mention_tag_nn_output_list(mention0)
        single_y_output += self.get_mention_tag_nn_output_list(mention1)

        return single_y_output

    def prepare_train_data(self, documents: typing.List[Document]):
        X = []
        y = []

        for document in documents:
            for i in range(len(document.tokens) - 1):
                token0 = document.tokens[i]
                token1 = document.tokens[i + 1]
                single_X_input = self.get_single_input(token0, token1)

                mention0 = self.get_mention_by_token(
                    document=document, token_index=token0.id
                )
                mention1 = self.get_mention_by_token(
                    document=document, token_index=token1.id
                )
                single_y_output = self.get_single_output(mention0, mention1)

                X.append(single_X_input)
                y.append(single_y_output)

        return X, y

    def get_token_postag_list(self, documents: typing.List[Document]):
        postag_list = []
        for document in documents:
            for token in document.tokens:
                postag_list.append(token.pos_tag)
        return list(set(postag_list))

    def get_mention_tag_list(self, documents: typing.List[Document]):
        tag_list = []
        for document in documents:
            for mention in document.mentions:
                tag_list.append(mention.tag)
        return list(set(tag_list))

    def get_token_postag_nn_input_list(self, token: Token):
        nn_input_list = []
        for postag in self.token_postag_list:
            if token.pos_tag == postag:
                nn_input_list.append(1)
            else:
                nn_input_list.append(0)
        return nn_input_list

    def get_mention_tag_nn_output_list(self, mention: Mention):
        nn_output_list = []
        if mention is None:
            return [0] * len(self.mention_tag_list)
        for tag in self.mention_tag_list:
            if mention.tag == tag:
                nn_output_list.append(1)
            else:
                nn_output_list.append(0)
        return nn_output_list

    def predict(
        self, content: str, schema: Schema, tokens: typing.List[Token]
    ) -> typing.List[CMention]:
        predictions = []

        for i in range(len(tokens) - 1):
            token0 = tokens[i]
            token1 = tokens[i + 1]

            input = self.get_single_input(token0, token1)
            input = torch.tensor([input], dtype=torch.float32)

            self.eval()
            output = self(input)
            predictions.append(output)

        threshold = 0.5

        ret = []
        mention = None
        for i, prediction in enumerate(predictions):
            if mention is None:
                if prediction[0, 0] > threshold:
                    mention = CMention(
                        endTokenDocumentIndex=i, startTokenDocumentIndex=i, type="TBC"
                    )
                    mention.startTokenDocumentIndex = i

            if mention is not None and prediction[0, 2] < threshold:
                mention.endTokenDocumentIndex = i
                ret.append(mention)
                mention = None

        return ret
