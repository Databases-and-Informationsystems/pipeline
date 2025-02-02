import os
import typing
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from abc import ABC, abstractmethod
from enum import Enum

from app.model.document import Document, Token, Mention, Relation
from app.model.schema import Schema
from app.model.settings import ModelSize
from app.util.llm_util import get_prediction, extract_json
from app.word_embeddings.word2vec import Word2VecModel


def get_token_postag_list(documents: typing.List[Document]):
    postag_list = []
    for document in documents:
        for token in document.tokens:
            postag_list.append(token.pos_tag)
    return list(set(postag_list))


def get_mention_tag_list(documents: typing.List[Document]):
    tag_list = []
    for document in documents:
        for mention in document.mentions:
            tag_list.append(mention.tag)
    return list(set(tag_list))


def get_relation_tag_list(documents: typing.List[Document]):
    tag_list = []
    for document in documents:
        for relation in document.relations:
            tag_list.append(relation.tag)
    return list(set(tag_list))


def get_entity_by_mention(document: Document, mention_index: int) -> Mention:
    for entity in document.entitys:
        for mention in entity.mentions:
            if mention.id == mention_index:
                return entity
    return


def cluster_pairs(pairs):
    clusters = []
    for a, b in pairs:
        overlapping = [cluster for cluster in clusters if a in cluster or b in cluster]

        if overlapping:
            merged_cluster = set(itertools.chain(*overlapping)) | {a, b}
            clusters = [c for c in clusters if c not in overlapping]
            clusters.append(merged_cluster)
        else:
            clusters.append({a, b})

    return [list(cluster) for cluster in clusters]


def get_min_max_token_indices_by_mention(
    mention: Mention,
) -> typing.Tuple[int, int]:
    token_indices = []
    for token in mention.tokens:
        token_indices.append(token.id)

    return min(token_indices), max(token_indices)


def get_relation_by_mentions(
    document: Document, head_mention_index: int, tail_mention_index: int
) -> Mention:
    for relation in document.relations:
        if (
            relation.head_mention.id == head_mention_index
            and relation.tail_mention.id == tail_mention_index
        ):
            return relation
    return
