from typing import List

import requests

from abc import ABC


class AccessOllama(ABC):
    """
    A simple wrapper for Ollama's API. The wrapper is not trying to be complete, just to provide a simple way to access
    the API. YOu can find more information about the API at https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(self, host: str = "localhost", port: int = 11434, protocol: str = "http"):
        self.connection = f"{protocol}://{host}:{port}"

    def list_models(self) -> List[str]:
        """
        List the available models on the provided host, usually you local machine.
        :return: List of strings with the available models in the format name (#params, quantization level).
        """

        response = requests.get(f"{self.connection}/api/tags")
        models = []
        if response.status_code == 200:
            json_response = response.json()
            for model in json_response["models"]:
                models.append(f"{model['name']} ( "
                              f"{model['details']['parameter_size']} - "
                              f"{model['details']['quantization_level']} )")
        return models

    def generate_answer(self, prompt: str, model: str) -> str:
        """
        Generate an answer to a prompt using the provided model.
        :param prompt: Complete prompt to send to the model.
        :param model: The model to use to generate the answer, has to be available in you Ollama instance.
        :return:
        """
        response = requests.post(f"{self.connection}/api/generate",
                                 json={
                                     "prompt": prompt,
                                     "model": model,
                                     "format": "json",
                                     "stream": False
                                 })
        if response.status_code == 200:
            return response.json()["response"]
        raise Exception("Error generating answer:" + response.text)

    def generate_embedding(self, text: str, model: str) -> List[float]:
        response = requests.post(f"{self.connection}/api/embeddings",
                                 json={"model": model, "prompt": text})
        if response.status_code == 200:
            return response.json()["embedding"]

        raise Exception("Error generating embedding:" + response.text)
