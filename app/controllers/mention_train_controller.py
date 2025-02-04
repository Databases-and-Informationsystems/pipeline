import json
import typing

from flask import request
from flask_restx import Resource
from pydantic import TypeAdapter

from . import train_nn_ns as ns
from ..model.document import Document
from ..model.schema import Schema
from ..model.settings import ModelSize
from ..train.factory import MentionTrainerFactory, get_mention_train_settings
from ..train.trainers.mention_trainer import MentionTrainer, MentionTrainModelType
from ..restx_dtos import train_nn_input, training_results, model_type_with_settings


@ns.route("/mention")
class MentionTrainController(Resource):

    @ns.doc(
        description="Defines all possible 'train'- _model_types_ with its possible _settings_ for the mention training step.",
    )
    @ns.marshal_with(model_type_with_settings, as_list=True, code=200)
    def get(self):
        return [
            {
                "model_type": model_type.value,
                "settings": get_mention_train_settings(model_type),
            }
            for model_type in MentionTrainModelType
        ]

    @ns.expect(train_nn_input, validate=True)
    @ns.marshal_with(training_results, code=200)
    @ns.doc(
        params={
            "model_type": {
                "description": f"Model type (default: {MentionTrainModelType.get_default().value})",
                "required": False,
                "enum": [model_type.value for model_type in MentionTrainModelType],
            },
            "model_size": {
                "description": f"Model size (default: {ModelSize.get_default().value})",
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

        training_results = mention_trainer.train(documents=documents, schema=schema)

        return training_results
