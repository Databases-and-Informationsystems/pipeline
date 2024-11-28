from flask_restx import Namespace, Resource

api = Namespace("test", description="Product-related operations")

test_entites = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]


@api.route("/")
class TestController(Resource):
    def get(self):
        """List all products"""
        return test_entites
