import os
import typing

import torch
import torch.nn as nn
import torch.optim as optim

from app.model.document import Document, Token, Mention
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
        self, size: ModelSize = ModelSize.MEDIUM, documents: typing.List[Document] = []
    ):
        self.size = size
        self.token_postag_list = self.get_token_postag_list(documents=documents)
        self.mention_tag_list = self.get_mention_tag_list(documents=documents)
        self.word2vec = Word2VecModel()

        super(MentionBasicNN, self).__init__()
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
        print("mention training started...")

        X, y = self.prepare_data(schema=schema, documents=documents)

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

        print("training finished")
        return "trained"

    def evaluate(self, schema: Schema, documents: typing.List[Document]) -> str:
        print("mention evaluated")
        return "evaluated"

    def save(self) -> bool:
        print("saved")
        return True

    def get_mention_by_token(self, document: Document, token_index: int):
        for mention in document.mentions:
            for token in mention.tokens:
                if token.id == token_index:
                    return mention
        return

    def save_as_file(self, file_name):
        directory = "mention_NNs"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f"{file_name}.pth")
        torch.save(self.state_dict(), file_path)

        print(f"Modell gespeichert unter: {file_path}")

    def prepare_data(self, schema: Schema, documents: typing.List[Document]):
        X = []
        y = []

        for document in documents:
            for i in range(len(document.tokens) - 1):
                token0 = document.tokens[i]
                token1 = document.tokens[i + 1]
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

                mention0 = self.get_mention_by_token(
                    document=document, token_index=token0.id
                )
                mention1 = self.get_mention_by_token(
                    document=document, token_index=token1.id
                )
                single_y_input = []

                if mention0:
                    single_y_input.append(1)
                else:
                    single_y_input.append(0)
                if mention1:
                    single_y_input.append(1)
                else:
                    single_y_input.append(0)

                if mention0 and mention1 and mention0.id == mention1.id:
                    single_y_input.append(1)
                else:
                    single_y_input.append(0)

                single_y_input += self.get_mention_tag_nn_output_list(mention0)
                single_y_input += self.get_mention_tag_nn_output_list(mention1)

                X.append(single_X_input)
                y.append(single_y_input)

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
