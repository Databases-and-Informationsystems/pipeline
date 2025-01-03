import typing

from flask import Blueprint, current_app
from flask_restx import Api, Namespace

main = Blueprint("main", __name__, url_prefix="/pipeline")
api = Api(
    main,
    title="Annotation Project Pipeline Microservice",
    version="1.0",
    description="A microservice to generate recommendations in different pipeline steps for tokens, mentions, relations and entities",
    doc="/docs",
    serve_path="/pipeline",  # available via /pipeline/docs
)


def caching_enabled():
    return current_app.config["CACHING"]


def get_document_id(data: any) -> typing.Optional[str]:
    document_id = None
    if caching_enabled():
        document_id = data.get("document_id")
        if document_id is None:
            raise ValueError("`document_id` is required with caching enabled")
    return document_id


# /steps/...
steps_ns: Namespace = Namespace("steps", description="Execute single pipeline steps")
from .tokenize_step_controller import TokenizeStepController
from .mention_step_controller import MentionStepController
from .entity_step_controller import EntityStepController
from .relation_step_controller import RelationStepController


api.add_namespace(steps_ns, path="/steps")
