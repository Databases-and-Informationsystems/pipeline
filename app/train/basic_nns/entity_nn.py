import typing

import torch
import torch.nn as nn
import torch.optim as optim
import itertools


from app.model.document import Document, CEntity, Mention, CMention
from app.model.schema import Schema
from app.model.settings import ModelSize
from app.util.llm_util import get_prediction, extract_json
from app.word_embeddings.word2vec import Word2VecModel
from app.train.basic_nns.basic_nn import BasicNN, BasicNNType


class EntityBasicNN(BasicNN):
    def __init__(
        self,
        size: ModelSize = ModelSize.MEDIUM,
        documents: typing.List[Document] = [],
        schema_id: typing.Optional[str] = None,
    ):
        super().__init__(
            nn_type=BasicNNType.ENTITY_NN,
            size=size,
            documents=documents,
            schema_id=schema_id,
        )

    def _init_layer(self):
        self.fc1 = nn.Linear(
            6
            + 2 * len(self.mention_tag_list)
            + 2 * len(self.token_postag_list)
            + 2 * self.word2vec.vector_size,
            100,
        )
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 30)
        self.fc4 = nn.Linear(30, 10)
        self.fc5 = nn.Linear(10, 1)

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

        wordvec0 = self.word2vec.get_vector(str0)
        wordvec1 = self.word2vec.get_vector(str1)

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

    def _prepare_train_data(self, documents: typing.List[Document]):
        X = []
        y = []

        for document in documents:
            for i in range(len(document.mentions)):
                for j in range(i + 1, len(document.mentions)):
                    mention0 = document.mentions[i]
                    mention1 = document.mentions[j]

                    single_X_input = self._get_single_input(mention0, mention1)
                    X.append(single_X_input)

                    entity0 = self._get_entity_by_mention(
                        document=document, mention_index=mention0.id
                    )
                    entity1 = self._get_entity_by_mention(
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

                input = self._get_single_input(mention0, mention1)
                input = torch.tensor([input], dtype=torch.float32)

                self.eval()
                output = self(input)
                predictions.append([[i, j], output])

        threshold = 0.5

        pairs = []

        for prediction in predictions:
            if prediction[1][0, 0] > threshold:
                mentionId0 = prediction[0][0]
                mentionId1 = prediction[0][1]
                pairs.append([mentionId0, mentionId1])

        cluster = self._cluster_pairs(pairs)
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

    def _get_entity_by_mention(self, document: Document, mention_index: int) -> Mention:
        for entity in document.entitys:
            for mention in entity.mentions:
                if mention.id == mention_index:
                    return entity
        return

    def _get_string_similarity(self, string0: str, string1: str) -> float:
        set0 = set(string0.lower())
        set1 = set(string1.lower())
        setIntersection = set0.intersection(set1)
        intersection = len(setIntersection)
        setUnion = set0.union(set1)
        union = len(setUnion)
        return intersection / union

    def _cluster_pairs(self, pairs):
        clusters = []
        for a, b in pairs:
            overlapping = [
                cluster for cluster in clusters if a in cluster or b in cluster
            ]

            if overlapping:
                merged_cluster = set(itertools.chain(*overlapping)) | {a, b}
                clusters = [c for c in clusters if c not in overlapping]
                clusters.append(merged_cluster)
            else:
                clusters.append({a, b})

        return [list(cluster) for cluster in clusters]
