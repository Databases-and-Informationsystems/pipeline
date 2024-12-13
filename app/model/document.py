import json
import typing
from enum import Enum

from pydantic import BaseModel


class Token(BaseModel):
    id: typing.Optional[int] = None
    text: typing.Optional[str] = None
    document_index: typing.Optional[int] = None
    sentence_index: typing.Optional[int] = None
    pos_tag: typing.Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "text": self.text,
                "document_index": self.document_index,
                "sentence_index": self.sentence_index,
                "pos_tag": self.pos_tag,
            }
        )


class Entity(BaseModel):
    id: int


class Mention(BaseModel):
    id: typing.Optional[int] = None
    tag: str
    tokens: typing.List[Token]
    entity: typing.Optional[Entity] = None

    def to_json(self, tokens: typing.List[Token]):
        tokens_with_content = list(
            map(
                lambda token: next((t for t in tokens if t.id == token.id), None),
                self.tokens,
            )
        )
        if any(t is None for t in tokens_with_content):
            raise ValueError(
                "The token of an mention is not part of the token list of the document"
            )
        sorted_tokens_with_content = sorted(
            tokens_with_content,
            key=lambda t: t.document_index,
        )

        return json.dumps(
            {
                "id": self.id,
                "tag": self.tag,
                "start_token_id": sorted_tokens_with_content[0].id,
                "end_token_id": sorted_tokens_with_content[-1].id,
                "text": " ".join(
                    list(map(lambda t: t.text, sorted_tokens_with_content))
                ),
            }
        )


class Relation(BaseModel):
    id: int
    tag: str
    head_mention: Mention
    tail_mention: Mention


class DocumentState(Enum):
    NEW = 1
    IN_PROGRESS = 2
    FINISHED = 3


class Document(BaseModel):
    id: typing.Optional[int] = None
    name: typing.Optional[str] = None
    content: str
    state: typing.Optional[DocumentState] = None
    tokens: typing.Optional[typing.List[Token]] = None


class DocumentEditState(Enum):
    MENTIONS = 1
    ENTITIES = 2
    RELATIONS = 3
    FINISHED = 4

    def to_dict(self):
        if self.MENTIONS:
            return "MENTIONS"
        elif self.ENTITIES:
            return "ENTITIES"
        elif self.RELATIONS:
            return "RELATIONS"
        elif self.FINISHED:
            return "FINISHED"


class DocumentEdit(BaseModel):
    document: Document
    state: typing.Optional[DocumentEditState] = None
    mentions: typing.Optional[typing.List[Mention]] = None
    relations: typing.Optional[typing.List[Relation]] = None
    entities: typing.Optional[typing.List[Entity]] = None

    class Config:
        from_attributes = True
