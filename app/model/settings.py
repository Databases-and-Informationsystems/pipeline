from enum import Enum


class GptModel(Enum):
    # TODO add other allowed models as enum states
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"

    @staticmethod
    def get_default():
        return GptModel.GPT_4O_MINI

    @staticmethod
    def from_string(value: str) -> "GptModel":
        """
        Map input value to equivalent enum state.
        If the input string is not represented by an enum state, return the default enum state.
        :param value:
        :return:
        """
        try:
            return GptModel(value)
        except ValueError:
            return GptModel.get_default()


class Temperature(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def to_float(self) -> float:
        match self:
            case Temperature.NONE:
                return 0.0
            case Temperature.LOW:
                return 0.2
            case Temperature.MEDIUM:
                return 0.7
            case Temperature.HIGH:
                return 1.2
        raise ValueError(f"Invalid temperature: {self}")

    @staticmethod
    def get_default():
        return Temperature.NONE

    @staticmethod
    def from_string(value: str) -> "Temperature":
        """
        Map input value to equivalent enum state.
        If the input string is not represented by an enum state, return the default enum state.
        :param value:
        :return:
        """
        try:
            return Temperature(value)
        except ValueError:
            return Temperature.get_default()


class ModelSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    BIG = "big"

    @staticmethod
    def get_default():
        return ModelSize.MEDIUM

    @staticmethod
    def from_string(value: str) -> "ModelSize":
        """
        Map input value to equivalent enum state.
        If the input string is not represented by an enum state, return the default enum state.
        :param value:
        :return:
        """
        try:
            return ModelSize(value)
        except ValueError:
            return ModelSize.get_default()
