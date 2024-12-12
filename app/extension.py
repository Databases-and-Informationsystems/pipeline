from flask import Blueprint
from flask_restx import Api

main = Blueprint("main", __name__, url_prefix="/pipeline")
api = Api(
    main,
    title="Annotation Project Pipeline Microservice",
    version="1.0",
    description="A microservice to generate recommendations in different pipeline steps for tokens, mentions, relations and entities",
    doc="/docs",
    serve_path="/pipeline",  # available via /pipeline/docs
)
