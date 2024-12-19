import json
import os
import typing

from app.pipeline.step import PipelineStepType

uploads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../uploads"))


def get_file_path(pipeline_step: PipelineStepType, document_id: str) -> str:
    return os.path.join(get_dir(pipeline_step), document_id)


def get_dir(pipeline_step: PipelineStepType) -> str:
    return os.path.join(uploads_dir, pipeline_step.value)


def create_file_from_data(data, pipeline_step: PipelineStepType, document_id: str):
    # Ensure the base uploads directory exists
    os.makedirs(uploads_dir, exist_ok=True)

    # Build the file path
    sub_dir = get_dir(pipeline_step)
    os.makedirs(sub_dir, exist_ok=True)  # Ensure the subdirectory exists

    file_path = get_file_path(pipeline_step, document_id)

    # Write the data to the file
    with open(file_path, "w+") as f:
        f.write(json.dumps(data))

    return file_path


def read_json_from_file(
    pipeline_step: PipelineStepType, document_id: str
) -> typing.Optional[any]:
    file_path = get_file_path(pipeline_step, document_id)
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
