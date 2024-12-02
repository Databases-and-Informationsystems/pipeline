from flask import Blueprint
from flask_restx import Api, Namespace

blueprint = Blueprint("pipeline", __name__)

api = Api(
    blueprint,
    title="Annotation Project Pipeline Microservice",
    version="1.0",
    description="A microservice to generate recommendations in different pipeline steps for tokens, mentions, relations and entites",
    doc="/docs",
    serve_path="/pipeline",  # available via /pipeline/docs
)

# Import and add namespaces
from .steps import steps_ns as steps_ns

api.add_namespace(steps_ns, path="/steps")  # /pipeline/steps
