from enum import Enum


class GptModel(Enum):
    # TODO add other allowed models as enum states
    GPT_4O_MINI = "gpt-4o-mini"

    @staticmethod
    def get_default():
        return GptModel.GPT_4O_MINI

    @staticmethod
    def from_string(value: str) -> "GptModel":
        if value == "gpt-4o-mini":
            return GptModel.GPT_4O_MINI
        else:
            raise ValueError(f"Unknown GPT model: {value}")


class Temperature(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def to_float(self) -> float:
        if self == Temperature.LOW:
            return 0.2
        if self == Temperature.MEDIUM:
            return 0.7
        if self == Temperature.HIGH:
            return 1.2
        return 0.0

    @staticmethod
    def get_default():
        return Temperature.NONE

    @staticmethod
    def from_string(value: str) -> "Temperature":
        value = value.lower()
        if value == "high":
            return Temperature.HIGH
        elif value == "medium":
            return Temperature.MEDIUM
        elif value == "low":
            return Temperature.LOW
        elif value == "none":
            return Temperature.NONE

        else:
            raise ValueError(f"Unknown temperature: {value}")


class PredictModelType(Enum):
    BASIC_NEURAL_NETWORK = "basic_nn"
    LARGE_LANGUAGE_MODEL = "llm"

    @staticmethod
    def get_default():
        return PredictModelType.LARGE_LANGUAGE_MODEL


class TrainModelType(Enum):
    BASIC_NEURAL_NETWORK = "basic_nn"

    @staticmethod
    def get_default():
        return TrainModelType.BASIC_NEURAL_NETWORK


class ModelSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    BIG = "big"

    @staticmethod
    def get_default():
        return ModelSize.MEDIUM

    @staticmethod
    def from_string(value: str) -> "ModelSize":
        value = value.lower()
        if value == "small":
            return ModelSize.SMALL
        elif value == "medium":
            return ModelSize.MEDIUM
        elif value == "big":
            return ModelSize.BIG

        else:
            raise ValueError(f"Unknown temperature: {value}")
