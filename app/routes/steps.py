from flask import jsonify
from flask_restx import Namespace, Resource

api = Namespace("steps", description="Execute single pipeline steps")


@api.route("/")
class PipelineStepController(Resource):
    def get(self):
        return jsonify({"general pipeline step": "Not implemented yet"})


@api.route("/tokenize")
class TokenizeStepController(Resource):
    def get(self):
        return jsonify({"Tokenize": "Not implemented yet"})

    @api.doc(
        description="Execute the tokenize step of the pipeline.",
        responses={
            200: ("Successful response", {}),
            500: "Internal Server Error",
        },
    )
    def post(self):
        return jsonify("Test Post Endpoint")
