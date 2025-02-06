import json
import typing

from flask import request, Response
from flask_restx import Resource
from pydantic import TypeAdapter

from . import train_nn_ns as ns
from ..model.document import Document
from ..model.schema import Schema
from ..model.settings import ModelSize
from ..train.factory import MentionTrainerFactory, get_mention_train_settings
from ..train.trainers.mention_trainer import MentionTrainer, MentionTrainModelType
from ..train.delete_service import delete_model
from ..train.basic_nns.basic_nn import BasicNNType
from ..restx_dtos import train_nn_input, training_results, model_type_with_settings
from ..util.logger import logger


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
            "name": {
                "description": "Name of the model. To get prediction from this you have to use the same name in step api",
                "required": True,
                "type": "string",
            },
        }
    )
    @ns.response(400, "Invalid input")
    @ns.response(500, "Internal server error")
    def post(self):
        """
        Train model for mention detection.
        """

        data = request.get_json()
        documents_data = data.get("documents")
        schema_data = data.get("schema")

        documents: typing.List[Document] = [
            TypeAdapter(Document).validate_json(json.dumps(document_data))
            for document_data in documents_data
        ]

        schema = TypeAdapter(Schema).validate_json(json.dumps(schema_data))

        mention_trainer: MentionTrainer = MentionTrainerFactory.create(request.args)

        training_results = mention_trainer.train(documents=documents, schema=schema)

        return training_results

    @ns.doc(
        description="Delete trained model for mention detection by given model type und model name.",
        params={
            "model_type": {
                "description": f"Model type (default: {MentionTrainModelType.get_default().value})",
                "required": False,
                "enum": [model_type.value for model_type in MentionTrainModelType],
            },
            "name": {
                "description": "Name of the model.",
                "required": True,
                "type": "string",
            },
        },
    )
    def delete(self):
        """
        Delete trained model for mention detection by given model type und model name
        """
        delete_model(request.args, BasicNNType.MENTION_NN)
        logger.info(
            f"Delete trained model for mention detection. {{'model_type'={request.args.get('model_type')}, 'name'={request.args.get('name')}}}"
        )
        return Response(status=204)
