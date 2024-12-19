import json
import typing
from enum import Enum

from pydantic import BaseModel


class CToken(BaseModel):
    text: typing.Optional[str] = None
    document_index: typing.Optional[int] = None
    sentence_index: typing.Optional[int] = None
    pos_tag: typing.Optional[str] = None


class Token(CToken):
    id: typing.Optional[int] = None

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


class CMention(BaseModel):
    type: str
    startTokenDocumentIndex: int
    endTokenDocumentIndex: int


class Mention(BaseModel):
    id: typing.Optional[int]
    tag: str
    tokens: typing.List[Token]
    entity: typing.Optional[Entity] = None

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        sorted_tokens_with_content = sorted(
            self.tokens,
            key=lambda t: t.document_index,
        )

        return {
            "id": self.id,
            "tag": self.tag,
            "start_token_id": sorted_tokens_with_content[0].id,
            "end_token_id": sorted_tokens_with_content[-1].id,
            "text": " ".join(list(map(lambda t: t.text, sorted_tokens_with_content))),
        }

    def to_json(self):
        return json.dumps(
            self.to_dict(),
        )


class CEntity(BaseModel):
    mentions: typing.List[Mention]

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "mentions": [mention.to_dict() for mention in self.mentions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


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
