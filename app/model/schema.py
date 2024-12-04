import typing

from pydantic import BaseModel


class SchemaMention(BaseModel):
    id: typing.Optional[int]
    tag: str
    description: str


class SchemaRelation(BaseModel):
    id: typing.Optional[int]
    tag: str


class SchemaConstraint(BaseModel):
    id: typing.Optional[int]
    schema_relation: SchemaRelation
    schema_mention_head: SchemaMention
    schema_mention_tail: SchemaMention
    is_directed: typing.Optional[bool]


class Schema(BaseModel):
    _id: typing.Optional[int]
    schema_mentions: typing.List[SchemaMention]
    _schema_relations: typing.List[SchemaRelation]
    _schema_constraints: typing.List[SchemaConstraint]
