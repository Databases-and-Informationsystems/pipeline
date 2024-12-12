from flask import Flask
from dotenv import load_dotenv


def create_app():
    app = Flask(__name__)
    load_dotenv()

    from .extension import main, api

    app.register_blueprint(main)
    from app.controllers import steps_ns as steps_ns
    from app.controllers.tokenize_step_controller import TokenizeStepController

    api.add_namespace(steps_ns, path="/steps")

    return app
