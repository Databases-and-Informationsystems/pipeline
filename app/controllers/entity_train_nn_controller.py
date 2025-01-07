import json
import typing

from flask import request
from flask_restx import Resource
from pydantic import TypeAdapter

from . import train_nn_ns as ns, caching_enabled, get_document_id
from ..model.document import Mention, CEntity, Document
from ..model.schema import Schema
from ..model.settings import ModelSize, TrainModelType
from ..train.factory import EntityTrainerFactory
from ..train.trainers.entity_trainer import EntityTrainer
from ..restx_dtos import train_entity_nn_input
from ..util.file import read_json_from_file, create_file_from_data


@ns.route("/entity")
class EntityTrainNNController(Resource):

    @ns.expect(train_entity_nn_input, validate=True)
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
        Train neural network for entity detection.
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

        entity_trainer: EntityTrainer = EntityTrainerFactory.create(
            settings={
                "model_size": request.args.get("model_size"),
                "model_type": request.args.get("model_type"),
                "evaluate": evaluate,
            }
        )

        entity_trainer.train(documents=documents, schema=schema)

        return "hallo"
