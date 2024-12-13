from enum import Enum


class GptModel(Enum):
    # TODO add other allowed models as enum states
    GPT_4O_MINI = "gpt-4o-mini"

    @staticmethod
    def from_string(value: str) -> "GptModel":
        if value == "gpt-4o-mini":
            return GptModel.GPT_4O_MINI
        else:
            raise ValueError(f"Unknown GPT model: {value}")
