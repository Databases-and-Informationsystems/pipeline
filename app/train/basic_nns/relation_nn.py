import typing

import torch
import numpy as np

from app.model.document import Document, CEntity, Mention, CRelation
from app.model.settings import ModelSize
from app.train.basic_nns.basic_nn import BasicNN, BasicNNType
from app.train.basic_nns.basic_nn_utils import get_relation_by_mentions
from app.util.logger import logger


class RelationBasicNN(BasicNN):
    def __init__(
        self,
        name: str,
        size: ModelSize = ModelSize.MEDIUM,
        documents: typing.List[Document] = [],
    ):
        super().__init__(
            nn_type=BasicNNType.RELATION_NN,
            size=size,
            name=name,
            documents=documents,
        )

    def _get_input_output_size(self):
        input_size = (
            6
            + 2 * len(self.mention_tag_list)
            + 2 * len(self.token_postag_list)
            + 2 * self.word2vec.vector_size * self.max_word_for_mention_vector
        )
        output_size = 1 + len(self.relation_tag_list)
        return input_size, output_size

    def _get_single_input(self, head_mention: Mention, tail_mention: Mention):
        single_X_input = []

        head_str = self._get_mention_text(head_mention)
        tail_str = self._get_mention_text(tail_mention)

        single_X_input.append(self._get_string_similarity(head_str, tail_str))

        single_X_input.append(head_mention.tokens[0].sentence_index)
        single_X_input.append(tail_mention.tokens[0].sentence_index)

        single_X_input += self._get_mention_tag_nn_input_list(head_mention)
        single_X_input += self._get_mention_tag_nn_input_list(tail_mention)

        single_X_input += self._get_mention_postag_nn_input_list(head_mention)
        single_X_input += self._get_mention_postag_nn_input_list(tail_mention)

        single_X_input.append(len(head_mention.tokens))
        single_X_input.append(len(tail_mention.tokens))

        index_distance = head_mention.id - tail_mention.id
        single_X_input.append(abs(index_distance))

        wordvecs0 = self.word2vec.get_multiple_vector_for_multiple_words(
            head_str.split(), self.max_word_for_mention_vector
        )
        wordvecs1 = self.word2vec.get_multiple_vector_for_multiple_words(
            tail_str.split(), self.max_word_for_mention_vector
        )

        for wordvec in wordvecs0:
            single_X_input.extend(wordvec)

        for wordvec in wordvecs1:
            single_X_input.extend(wordvec)

        return single_X_input

    def _get_single_output(
        self, document: Document, head_mention: Mention, tail_mention: Mention
    ):
        single_y_output = []

        relation = get_relation_by_mentions(
            document=document,
            head_mention_index=head_mention.id,
            tail_mention_index=tail_mention.id,
        )

        if relation is None:
            single_y_output.append(1)
        else:
            single_y_output.append(0)

        single_y_output += self._get_relation_tag_nn_input_list(relation)

        return single_y_output

    def _prepare_train_data(self, documents: typing.List[Document]):
        X = []
        y = []

        for document in documents:
            for i in range(len(document.mentions)):
                for j in range(i + 1, len(document.mentions)):
                    for k in range(2):
                        if k == 0:
                            head_mention = document.mentions[i]
                            tail_mention = document.mentions[j]
                        else:
                            head_mention = document.mentions[j]
                            tail_mention = document.mentions[i]

                    X.append(self._get_single_input(head_mention, tail_mention))
                    y.append(
                        self._get_single_output(
                            document=document,
                            head_mention=head_mention,
                            tail_mention=tail_mention,
                        )
                    )

        return X, y

    def predict(self, mentions: typing.List[Mention]) -> typing.List[CEntity]:
        predictions = []

        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                for k in range(2):
                    if k == 0:
                        head_mention_id = i
                        tail_mention_id = j
                    else:
                        head_mention_id = j
                        tail_mention_id = i

                head_mention = mentions[head_mention_id]
                tail_mention = mentions[tail_mention_id]

                input = self._get_single_input(head_mention, tail_mention)
                input = torch.tensor([input], dtype=torch.float32)

                self.eval()
                output = self(input)
                predictions.append([[i, j], output])

        threshold = 0.5

        relations: typing.List[CRelation] = []

        for prediction in predictions:
            if prediction[1][0, 0] > threshold:
                head_mention_id = prediction[0][0]
                tail_mention_id = prediction[0][1]

                tag_prediction = prediction[1][0][1:]
                tag_index = torch.argmax(tag_prediction)
                tag = self.relation_tag_list[tag_index]

                c_relation = CRelation(
                    head_mention_id=head_mention_id,
                    tail_mention_id=tail_mention_id,
                    tag=tag,
                )
                relations.append(c_relation)

        logger.debug(f"Relation basic nn prediction output:\n{relations}")
        return relations

    def _evaluate_prediction_against_truth(
        self, prediction: typing.List[CRelation], truth: Document
    ):
        counter = 0
        for prediciton_relation in prediction:
            for true_relation in truth.relations:
                if (
                    prediciton_relation.head_mention_id == true_relation.head_mention.id
                    and prediciton_relation.tail_mention_id
                    == true_relation.tail_mention.id
                    # and prediciton_relation.tag == true_relation.tag
                ):
                    counter += 1
        return counter * 2 / (len(truth.mentions) + len(prediction))
