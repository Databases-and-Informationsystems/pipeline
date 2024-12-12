from flask import Flask
from dotenv import load_dotenv


def create_app():
    app = Flask(__name__)
    load_dotenv()

    from .controllers import main

    app.register_blueprint(main)

    return app
