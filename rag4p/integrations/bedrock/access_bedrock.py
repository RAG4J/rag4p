import json
from abc import ABC
from typing import List

import boto3
from dotenv import load_dotenv

from rag4p.util.key_loader import KeyLoader


class AccessBedrock(ABC):

    def __init__(self, region_name: str = 'eu-central-1'):
        self.region_name = region_name

        self.bedrock = boto3.client(service_name='bedrock', region_name=self.region_name)
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=self.region_name)

    def list_models(self):
        response = self.bedrock.list_foundation_models()
        models = []
        for model in response['modelSummaries']:
            models.append({
                "id": model['modelId'],
                "name": model['modelName'],
                "provider": model['providerName'],
                "output_modalities": model['outputModalities'],
            })
        return models

    def generate_answer(self, prompt: str, model: str) -> str:
        """
        Generate an answer to a prompt using the provided model.
        :param prompt: Complete prompt to send to the model.
        :param model: The model to use to generate the answer, has to be available in you Ollama instance.
        :return:
        """
        # Inference parameters to use.
        temperature = 0.5
        top_p = 1

        # Base inference parameters to use.
        inference_config = {"temperature": temperature, "topP": top_p}

        # Additional inference parameters to use.
        additional_model_fields = {}

        messages = [
            {
                "role": "user",
                "content": [{
                    "text": prompt
                }]
            }
        ]
        # Send the message.
        bedrock_response = self.bedrock_runtime.converse(
            modelId=model,
            messages=messages,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )

        # token_usage = bedrock_response['usage']
        # print("Input tokens: %s", token_usage['inputTokens'])
        # print("Output tokens: %s", token_usage['outputTokens'])
        # print("Total tokens: %s", token_usage['totalTokens'])
        # print("Stop reason: %s", bedrock_response['stopReason'])

        return bedrock_response['output']['message']['content'][0]['text']

    def generate_embedding(self, text: str, model: str) -> List[float]:
        accept = "application/json"
        content_type = "application/json"

        if model.startswith("cohere."):
            body = json.dumps({
                "texts": [text],
                "input_type": "search_document",
            })
        else:
            body = json.dumps({
                "inputText": text,
            })

        response = self.bedrock_runtime.invoke_model(
            body=body, modelId=model, accept=accept, contentType=content_type
        )

        response_body = json.loads(response.get('body').read())

        finish_reason = response_body.get("message")
        if finish_reason is not None:
            raise ValueError(f"Embeddings generation error: {finish_reason}")

        if model.startswith("cohere."):
            return response_body["embeddings"][0]
        else:
            return response_body["embedding"]

    @staticmethod
    def init_from_env(key_loader: KeyLoader):
        region_name = key_loader.get_bedrock_region()
        return AccessBedrock(region_name=region_name)


if __name__ == "__main__":
    load_dotenv()

    access_bedrock = AccessBedrock.init_from_env(KeyLoader())
    models_ = access_bedrock.list_models()
    print(models_)

    text = "What is the capital of Germany, is it the same as when Germany was devided into East and West?"
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    response_ = access_bedrock.generate_answer(text, model_id)
    print(response_)

    text = "What is the capital of Germany, is it the same as when Germany was devided into East and West?"
    model_id = "cohere.embed-english-v3"
    embedding = access_bedrock.generate_embedding(text, model_id)
    print(f"Embedding length: {len(embedding)}")
