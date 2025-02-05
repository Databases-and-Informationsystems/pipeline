import typing

import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np

from app.train.basic_nns.basic_nn_utils import get_entity_by_mention, cluster_pairs
from app.model.document import Document, CEntity, Mention, CMention
from app.model.schema import Schema
from app.model.settings import ModelSize
from app.util.llm_util import get_prediction, extract_json
from app.word_embeddings.word2vec import Word2VecModel
from app.train.basic_nns.basic_nn import BasicNN, BasicNNType


class EntityBasicNN(BasicNN):
    max_word_for_mention_vector: int

    def __init__(
        self,
        name: str,
        size: ModelSize = ModelSize.MEDIUM,
        documents: typing.List[Document] = [],
        schema_id: typing.Optional[str] = None,
        max_word_for_mention_vector=3,
    ):
        self.max_word_for_mention_vector = max_word_for_mention_vector
        super().__init__(
            nn_type=BasicNNType.ENTITY_NN,
            size=size,
            name=name,
            documents=documents,
            schema_id=schema_id,
        )

    def _get_input_output_size(self):
        input_size = (
            6
            + 2 * len(self.mention_tag_list)
            + 2 * len(self.token_postag_list)
            + 2 * self.word2vec.vector_size * self.max_word_for_mention_vector
        )
        output_size = 1
        return input_size, output_size

    def _get_single_input(self, mention0: Mention, mention1: Mention):
        single_X_input = []

        str0 = self._get_mention_text(mention0)
        str1 = self._get_mention_text(mention1)

        single_X_input.append(self._get_string_similarity(str0, str1))

        single_X_input.append(mention0.tokens[0].sentence_index)
        single_X_input.append(mention1.tokens[0].sentence_index)

        single_X_input += self._get_mention_tag_nn_input_list(mention0)
        single_X_input += self._get_mention_tag_nn_input_list(mention1)

        single_X_input += self._get_mention_postag_nn_input_list(mention0)
        single_X_input += self._get_mention_postag_nn_input_list(mention1)

        single_X_input.append(len(mention0.tokens))
        single_X_input.append(len(mention1.tokens))

        index_distance = mention0.id - mention1.id
        single_X_input.append(abs(index_distance))

        wordvecs0 = self.word2vec.get_multiple_vector_for_multiple_words(
            str0.split(), self.max_word_for_mention_vector
        )
        wordvecs1 = self.word2vec.get_multiple_vector_for_multiple_words(
            str1.split(), self.max_word_for_mention_vector
        )

        for wordvec in wordvecs0:
            single_X_input.extend(wordvec)

        for wordvec in wordvecs1:
            single_X_input.extend(wordvec)

        return single_X_input

    def _prepare_train_data(self, documents: typing.List[Document]):
        X = []
        y = []

        for document in documents:
            for i in range(len(document.mentions)):
                for j in range(i + 1, len(document.mentions)):
                    mention0 = document.mentions[i]
                    mention1 = document.mentions[j]

                    if mention0.tag != mention1.tag:
                        continue

                    single_X_input = self._get_single_input(mention0, mention1)
                    X.append(single_X_input)

                    entity0 = get_entity_by_mention(
                        document=document, mention_index=mention0.id
                    )
                    entity1 = get_entity_by_mention(
                        document=document, mention_index=mention0.id
                    )

                    if entity0 and entity1 and entity0.id == entity1.id:
                        y.append([1])
                    else:
                        y.append([0])

        return X, y

    def predict(self, mentions: typing.List[Mention]) -> typing.List[CEntity]:
        predictions = []

        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                mention0 = mentions[i]
                mention1 = mentions[j]

                if mention0.tag != mention1.tag:
                    continue

                input = self._get_single_input(mention0, mention1)
                input = torch.tensor([input], dtype=torch.float32)

                self.eval()
                output = self(input)
                predictions.append([[i, j], output])

        threshold = 0.7

        pairs = []

        for prediction in predictions:
            if prediction[1][0, 0] > threshold:
                mentionId0 = prediction[0][0]
                mentionId1 = prediction[0][1]
                pairs.append([mentionId0, mentionId1])

        cluster = cluster_pairs(pairs)
        entitys: typing.List[CEntity] = []
        for mention_ids in cluster:
            pred_mentions: typing.List[Mention] = []
            for mention_id in mention_ids:
                pred_mentions.append(mentions[mention_id])
            entitys.append(CEntity(mentions=pred_mentions))

        return entitys

    def _evaluate_prediction_against_truth(
        self, prediction: typing.List[CEntity], truth: Document
    ):
        counter = 0
        for prediciton_entity in prediction:
            pred_index_list = []
            for mention in prediciton_entity.mentions:
                pred_index_list.append(mention.id)

            for true_entity in truth.entitys:
                true_index_list = []
                for mention in true_entity.mentions:
                    true_index_list.append(mention.id)

                if set(pred_index_list) == set(true_index_list):
                    counter += 1
        return counter * 2 / (len(truth.mentions) + len(prediction))
