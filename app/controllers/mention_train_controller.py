import json
import typing

from flask import request
from flask_restx import Resource
from pydantic import TypeAdapter

from . import train_nn_ns as ns, caching_enabled, get_document_id
from ..model.document import Mention, CEntity, Document
from ..model.schema import Schema
from ..model.settings import ModelSize, TrainModelType
from ..pipeline.factory import EntityStepFactory
from ..train.factory import MentionTrainerFactory
from ..pipeline.steps.entity_prediction import EntityStep
from ..train.trainers.mention_trainer import MentionTrainer
from ..restx_dtos import train_entity_input
from ..util.file import read_json_from_file, create_file_from_data


@ns.route("/mention")
class MentionTrainController(Resource):

    @ns.expect(train_entity_input, validate=True)
    @ns.doc(
        params={
            "model_type": {
                "description": f"Model size (default: {TrainModelType.get_default().value})",
                "required": False,
                "enum": [model_size.value for model_size in TrainModelType],
            },
            "model_size": {
                "description": f"Model type (default: {ModelSize.get_default().value})",
                "required": False,
                "enum": [model_size.value for model_size in ModelSize],
            },
            "enable_evaluation": {
                "description": f"Enable evaluation (5-fold cross validation)",
                "required": False,
                "type": "bool",
            },
        }
    )
    @ns.response(400, "Invalid input")
    @ns.response(500, "Internal server error")
    def post(self):
        """
        Train neural network for mention detection.
        """

        data = request.get_json()
        documents_data = data.get("documents")
        schema_data = data.get("schema")

        documents: typing.List[Document] = [
            TypeAdapter(Document).validate_json(json.dumps(document_data))
            for document_data in documents_data
        ]

        schema = TypeAdapter(Schema).validate_json(json.dumps(schema_data))
        evaluate = request.args.get("enable_evaluation", "false").lower() == "true"

        mention_trainer: MentionTrainer = MentionTrainerFactory.create(
            settings={
                "model_size": request.args.get("model_size"),
                "model_type": request.args.get("model_type"),
                "evaluate": evaluate,
            }
        )

        mention_trainer.train(documents=documents, schema=schema)

        return "hallo"
