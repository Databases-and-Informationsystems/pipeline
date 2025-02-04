import typing

import torch
import numpy as np

from app.train.basic_nns.basic_nn_utils import get_min_max_token_indices_by_mention
from app.model.document import Document, Token, Mention, CMention
from app.model.settings import ModelSize
from app.train.basic_nns.basic_nn import BasicNN, BasicNNType


class MentionBasicNN(BasicNN):
    token_postag_list: list[str]
    mention_tag_list: list[str]

    def __init__(
        self,
        name: str,
        size: ModelSize = ModelSize.MEDIUM,
        documents: typing.List[Document] = [],
        schema_id: typing.Optional[str] = None,
    ):
        self.token_postag_list = self._get_token_postag_list(documents=documents)
        self.mention_tag_list = self._get_mention_tag_list(documents=documents)
        super().__init__(
            nn_type=BasicNNType.MENTION_NN,
            size=size,
            name=name,
            documents=documents,
            schema_id=schema_id,
        )

    def _get_input_output_size(self):
        input_size = 4 + 2 * len(self.token_postag_list) + 2 * self.word2vec.vector_size
        output_size = 3 + 2 * len(self.mention_tag_list)
        return input_size, output_size

    def _get_mention_by_token(self, document: Document, token_index: int):
        for mention in document.mentions:
            for token in mention.tokens:
                if token.id == token_index:
                    return mention
        return

    def _get_single_input(self, token0: Token, token1: Token):
        single_X_input = []

        single_X_input.append(token0.document_index)
        single_X_input.append(token1.document_index)

        single_X_input.append(token0.sentence_index)
        single_X_input.append(token1.sentence_index)

        single_X_input += self._get_token_postag_nn_input_list(token0)
        single_X_input += self._get_token_postag_nn_input_list(token1)

        wordvec0 = self.word2vec.get_vector(token0.text)
        wordvec1 = self.word2vec.get_vector(token1.text)

        vector_size = self.word2vec.vector_size

        if wordvec0 is None or np.isnan(wordvec0).all():
            single_X_input.extend([0] * vector_size)
        else:
            single_X_input.extend(wordvec0)

        if wordvec1 is None or np.isnan(wordvec1).all():
            single_X_input.extend([0] * vector_size)
        else:
            single_X_input.extend(wordvec1)

        return single_X_input

    def _get_single_output(self, mention0: Mention, mention1: Mention):
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

        single_y_output += self._get_mention_tag_nn_input_list(mention0)
        single_y_output += self._get_mention_tag_nn_input_list(mention1)

        return single_y_output

    def _prepare_train_data(self, documents: typing.List[Document]):
        X = []
        y = []

        for document in documents:
            for i in range(len(document.tokens) - 1):
                token0 = document.tokens[i]
                token1 = document.tokens[i + 1]
                single_X_input = self._get_single_input(token0, token1)

                mention0 = self._get_mention_by_token(
                    document=document, token_index=token0.id
                )
                mention1 = self._get_mention_by_token(
                    document=document, token_index=token1.id
                )
                single_y_output = self._get_single_output(mention0, mention1)

                X.append(single_X_input)
                y.append(single_y_output)

        return X, y

    def predict(self, tokens: typing.List[Token]) -> typing.List[CMention]:
        predictions = []

        for i in range(len(tokens) - 1):
            token0 = tokens[i]
            token1 = tokens[i + 1]

            input = self._get_single_input(token0, token1)
            input = torch.tensor([input], dtype=torch.float32)

            self.eval()
            output = self(input)
            predictions.append(output)

        threshold = 0.5

        mentions: typing.List[CMention] = []
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
                mentions.append(mention)
                mention = None

        output_neuron_const = 3

        for mention in mentions:
            type_prediction = torch.zeros(
                int((predictions[0].shape[1] - output_neuron_const) / 2)
            )

            for i in range(
                mention.startTokenDocumentIndex - 1, mention.endTokenDocumentIndex + 1
            ):
                if i < 0 or i > len(prediction):
                    continue

                if i <= mention.endTokenDocumentIndex:
                    type_prediction += predictions[i][0][
                        output_neuron_const : output_neuron_const + len(type_prediction)
                    ]

                if i >= mention.startTokenDocumentIndex:
                    type_prediction += predictions[i][0][
                        output_neuron_const
                        + len(type_prediction) : output_neuron_const
                        + 2 * len(type_prediction)
                    ]

            max_index = torch.argmax(type_prediction)
            mention.type = self.mention_tag_list[max_index]

        return mentions

    def _evaluate_prediction_against_truth(
        self, prediction: typing.List[CMention], truth: Document
    ):
        truth_cmentions: typing.List[CMention] = []
        for truth_entry in truth.mentions:
            start, end = get_min_max_token_indices_by_mention(truth_entry)

            cmention = CMention(
                endTokenDocumentIndex=start,
                startTokenDocumentIndex=end,
                type=truth_entry.tag,
            )
            truth_cmentions.append(cmention)

        equal_counter = 0
        for prediciton_mention in prediction:
            for true_mention in truth_cmentions:
                if (
                    prediciton_mention.startTokenDocumentIndex
                    == true_mention.startTokenDocumentIndex
                    and prediciton_mention.endTokenDocumentIndex
                    == true_mention.endTokenDocumentIndex
                    and prediciton_mention.type == true_mention.type
                ):
                    equal_counter += 1
        return equal_counter * 2 / (len(truth_cmentions) + len(prediction))
