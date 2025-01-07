import typing
from dataclasses import field

from pydantic import BaseModel


class SchemaMention(BaseModel):
    id: typing.Optional[int] = None
    tag: str
    description: str
    has_entities: bool = False


class SchemaRelation(BaseModel):
    id: typing.Optional[int] = None
    tag: str
    description: str


class SchemaConstraint(BaseModel):
    id: typing.Optional[int] = None
    schema_relation: SchemaRelation
    schema_mention_head: SchemaMention
    schema_mention_tail: SchemaMention
    is_directed: typing.Optional[bool]


class Schema(BaseModel):
    id: typing.Optional[int] = None
    schema_mentions: typing.List[SchemaMention] = field(default_factory=list)
    schema_relations: typing.List[SchemaRelation] = field(default_factory=list)
    schema_constraints: typing.List[SchemaConstraint] = field(default_factory=list)
