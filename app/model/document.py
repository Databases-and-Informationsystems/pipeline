import typing
from enum import Enum

from pydantic import BaseModel


class Token(BaseModel):
    id: typing.Optional[int] = None
    text: str
    document_index: int
    sentence_index: int
    pos_tag: str


class Entity(BaseModel):
    id: int


class Mention(BaseModel):
    tag: str
    tokens: typing.List[Token]
    entity: typing.Optional[Entity] = None


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
    name: str
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
        orm_mode = True
