"""
title: AWS Bedrock Manifold Pipeline
author: SSS
date: 2024-10-26
version: 1.0
license: MIT
description: A pipeline for generating text using AWS Bedrock models in Open-WebUI.
requirements: boto3
environment_variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
"""

from typing import List, Union, Iterator
import os
import json
import boto3
import base64
from pydantic import BaseModel
import traceback

class Pipeline:
    """AWS Bedrock pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""
        AWS_ACCESS_KEY_ID: str
        AWS_SECRET_ACCESS_KEY: str
        AWS_REGION: str

    def __init__(self):
        self.type = "manifold"
        self.id = "aws_bedrock"
        self.name = "AWS Bedrock: "

        self.valves = self.Valves(
            **{
                "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "your-aws-access-key"),
                "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "your-aws-secret-key"),
                "AWS_REGION": os.getenv("AWS_REGION", "us-west-2"),
            }
        )
        self.pipelines = []

        # Initialize the Boto3 client for Bedrock and runtime
        self.client = boto3.client(
            "bedrock",
            region_name=self.valves.AWS_REGION,
            aws_access_key_id=self.valves.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.valves.AWS_SECRET_ACCESS_KEY
        )
        self.runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=self.valves.AWS_REGION,
            aws_access_key_id=self.valves.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.valves.AWS_SECRET_ACCESS_KEY
        )
        
        self.update_pipelines()

    async def on_startup(self) -> None:
        """Called when the server is started."""
        print(f"on_startup: {__name__}")

    async def on_shutdown(self) -> None:
        """Called when the server is stopped."""
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self) -> None:
        """Called when the valves are updated."""
        print(f"on_valves_updated: {__name__}")
        self.client = boto3.client(
            "bedrock",
            region_name=self.valves.AWS_REGION,
            aws_access_key_id=self.valves.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.valves.AWS_SECRET_ACCESS_KEY
        )
        self.runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=self.valves.AWS_REGION,
            aws_access_key_id=self.valves.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.valves.AWS_SECRET_ACCESS_KEY
        )
        self.update_pipelines()

    def update_pipelines(self) -> None:
        """Update available models from AWS Bedrock."""
        try:
            models = self.client.list_foundation_models(byInferenceType='ON_DEMAND', byOutputModality='TEXT')["modelSummaries"]
            self.pipelines = [
                {"id": model["modelId"], "name": model["modelName"]}
                for model in models
            ]
        except Exception as e:
            print(f"Failed to fetch models: {e}")
            self.pipelines = [
                {
                    "id": "error",
                    "name": "Could not fetch models from AWS Bedrock, please check your AWS credentials and region configuration.",
                }
            ]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        try:
            # Format the request body based on the model and messages
            payload = self._format_message(model_id, messages, body)

            accept = "application/json"
            content_type = "application/json"
            # Known issue - streaming doesn't work due to invoke_model_with_response_stream compatibility. 
            # if body.get("stream", False): 
            #     response = self.runtime_client.invoke_model_with_response_stream(
            #         body=payload.encode("utf-8"),
            #         modelId=model_id,
            #         accept=accept,
            #         contentType=content_type,
            #     )
            #     return self._stream_response(response, model_id.split(".")[0])
            # else:
            response = self.runtime_client.invoke_model(
                body=payload.encode("utf-8"),
                modelId=model_id,
                accept=accept,
                contentType=content_type,
            )
            response_body = json.loads(response.get("body").read())
            return self._extract_output(response_body, model_id)

        except Exception as e:
            print(f"Error generating content: {e}")
            traceback.print_exc()
            return f"An error occurred: {str(e)}"

    def _format_message(self, model: str, messages: List[dict], body: dict) -> str:
        """Format the message based on the model and list of messages."""
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.5)

        if 'amazon' in model:
            prompt = ""
            for message in messages:
                if message["role"] == "user":
                    prompt += f"{message['content']} "
            return json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "stopSequences": [],
                    "temperature": temperature
                }
            })
        elif 'anthropic' in model:
            # Handle image if present in the last user message
            content = []
            if messages and messages[-1]["role"] == "user" and body.get("image_url"):
                image_url = body["image_url"]
                try:
                    import requests
                    from io import BytesIO

                    response = requests.get(image_url)
                    response.raise_for_status()
                    image_data = base64.b64encode(response.content).decode('utf-8')
                    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}})
                except requests.exceptions.RequestException as e:
                    print(f"Error: Could not download image: {e}")
            
            # Construct messages for Anthropic
            anthropic_messages = []
            for message in messages:
                if message["role"] == "user":
                    content.append({"type": "text", "text": message["content"]})
                    anthropic_messages.append({"role": "user", "content": content})
                elif message["role"] == "assistant":
                    anthropic_messages.append({"role": "assistant", "content": message["content"]})

            return json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": anthropic_messages,
                "temperature": temperature,
            })
        elif 'ai21' in model:
            prompt = ""
            for message in messages:
                prompt += f"{message['role']}: {message['content']}\n"
            return json.dumps({
                "prompt": prompt,
                "temperature": temperature,
                "maxTokens": max_tokens,
            })
        elif 'cohere' in model:
            prompt = ""
            for message in messages:
                prompt += f"{message['role']}: {message['content']}\n"
            return json.dumps({
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            })
        elif 'meta' in model:
            prompt = ""
            for message in messages:
                prompt += f"{message['role']}: {message['content']}\n"
            return json.dumps({
                "prompt": prompt,
                "temperature": temperature,
                "max_gen_len": max_tokens
            })
        elif 'mistral' in model:
            prompt = ""
            for message in messages:
                if message["role"] == "user":
                    prompt += f"[INST] {message['content']} [/INST]"
                elif message["role"] == "assistant":
                    prompt += f"{message['content']}"
            return json.dumps({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
        else:
            return ""

    def _extract_output(self, jsondata: dict, model: str) -> str:
        """Extract the output from the model response."""
        if 'amazon' in model:
            return jsondata['results'][0]['outputText']
        elif 'anthropic' in model:
            return jsondata['content'][0]['text']
        elif 'ai21' in model:
            return jsondata.get('completions')[0].get('data').get('text')
        elif 'cohere' in model:
            return jsondata['generations'][0]['text']
        elif 'meta' in model:
            return jsondata['generation']
        elif 'mistral' in model:
            return jsondata['outputs'][0]['text']
        else:
            return ""

    def _stream_response(self, response, model_provider: str):
        """Stream the response chunks."""
        if model_provider == "anthropic":
            for event in response["body"]:
                data = json.loads(event["chunk"]["bytes"])
                yield data.get("completion", "")
        elif model_provider == "mistral":
            event_stream = response.get('body')
            for b in iter(event_stream):
                bc = b['chunk']['bytes']
                gen = json.loads(bc.decode('utf-8'))
                line = gen.get('outputs', [{}])[0].get('text', "")
                if '\n' == line:
                    yield ''
                else:
                    yield line
        elif model_provider == "cohere":
            for event in response["body"]:
                data = json.loads(event["chunk"]["bytes"])
                yield data.get('text', "")
        elif model_provider == "meta":
            for event in response["body"]:
                data = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                yield data.get('generation', "")
        else:  # Default streaming for other models
            for event in response["body"]:
                chunk = event["chunk"]["bytes"].decode("utf-8")
                yield chunk
