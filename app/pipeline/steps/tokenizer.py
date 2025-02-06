import typing
from abc import abstractmethod, ABC

import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

from app.model.document import Token, CToken
from app.pipeline.step import PipelineStep, PipelineStepType
from app.util.logger import logger


class TokenizeStep(PipelineStep, ABC):

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name, PipelineStepType.TOKENIZER)

    def run(self, content: str) -> typing.List[CToken]:
        logger.info(
            f"{self.name} with settings: {self._get_settings().__str__()} executed"
        )
        res = self._run(content)
        return res

    @abstractmethod
    def _run(self, content: str) -> typing.List[CToken]:
        pass


class Tokenizer(TokenizeStep):
    def __init__(self, name: str = "Tokenizer"):
        super().__init__(name)

        # Both 'punkt_tab' and 'averaged_perceptron_tagger_eng' are required for nltk tokenizing
        try:
            nltk.data.find("punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")

        try:
            nltk.data.find("averaged_perceptron_tagger_eng")
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng")

    def _run(self, content: str) -> typing.List[CToken]:
        tokenizer = PunktSentenceTokenizer()

        # Tokens where a "." does not mean the end of a sentence
        tokenizer._params.abbrev_types.update(
            ["e.g", "etc", "dr", "vs", "mr", "mrs", "prof", "inc", "i.e"]
        )
        sentences = tokenizer.tokenize(
            # ".." Is not a valid token but rather a shortcut + end of sentence
            content.replace("..", ". .")
        )

        tokens: typing.List[Token] = []
        document_index = 0
        sentence_index = 0

        for sentence in sentences:
            sentence_tokens = word_tokenize(sentence)
            tagged_tokens = nltk.pos_tag(sentence_tokens)

            for token, tag in tagged_tokens:
                tokens.append(
                    CToken(
                        text=token,
                        pos_tag=tag,
                        document_index=document_index,
                        sentence_index=sentence_index,
                    )
                )
                document_index += 1

            sentence_index += 1
        return tokens

    def _get_settings(self) -> typing.Dict[str, typing.Any]:
        return {}
