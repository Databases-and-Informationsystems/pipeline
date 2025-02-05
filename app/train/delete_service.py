import typing
import shutil
import os

from app.train.basic_nns.basic_nn import BasicNNType


def delete_model(settings: typing.Optional[dict], step_type: BasicNNType):
    model_type = settings.get("model_type")
    nn_name = settings.get("name")

    if not model_type or not nn_name:
        raise ValueError("Invalid settings. 'model_type' or 'name' are missing.")

    path = f"models/{model_type}/{step_type.value}/{nn_name}"

    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        raise ValueError(f"Model '{path}' does not exist.")
    return
