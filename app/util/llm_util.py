import openai

from app.pipeline.models.llm import GptModel


def get_prediction(prompt: str, model: GptModel, key: str, temperature: float = 0.0):
    openai.api_key = key
    response = openai.chat.completions.create(
        model=model.value,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a nlp expert."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def extract_json(prediction: str) -> str:
    start_idx_c = prediction.find("{")
    start_idx_s = prediction.find("[")
    end_idx_c = prediction.rfind("}")
    end_idx_s = prediction.rfind("]")

    # JSON Starts with { or [
    # We assume, that an ending bracket exits if a starting bracket exists
    if start_idx_c != -1 and start_idx_c < start_idx_s:
        start_idx = start_idx_c
        end_idx = end_idx_c
    else:
        start_idx = start_idx_s
        end_idx = end_idx_s

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return prediction[start_idx : end_idx + 1].strip()
    else:
        raise ValueError("The response contains no valid JSON structure.")
