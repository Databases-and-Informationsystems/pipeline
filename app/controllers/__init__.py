from flask import Blueprint
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

# /steps/...
steps_ns: Namespace = Namespace("steps", description="Execute single pipeline steps")
from .tokenize_step_controller import TokenizeStepController
from .mention_step_controller import MentionStepController
from .entity_step_controller import EntityStepController
from .relation_step_controller import RelationStepController


api.add_namespace(steps_ns, path="/steps")
