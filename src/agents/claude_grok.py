import anthropic
from openai import OpenAI
import os
import base64
import mimetypes
import re
class VisionExperiment:
    def __init__(self, model, prompt_func):
        self.model = model
        self.prompt_func = prompt_func
    def convert_to_base64(self, img_path):
        with open(img_path, "rb") as img_file:
            converted_img_data = base64.b64encode(img_file.read()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(img_path)
        return converted_img_data, mime_type

class ClaudeExperiment(VisionExperiment):
    def __init__(self, model, prompt_func):
        super().__init__(model, prompt_func)
        self.client = anthropic.Anthropic(api_key=os.getenv('CLAUDE'))
    def process_sample(self, img_path, tool, annotation_type):
        converted_img_data, mime_type = self.convert_to_base64(img_path=img_path)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[
            {
            "role": "user",
            "content": [
                    {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": converted_img_data,
                        },
                    },
                    {
                    "type": "text",
                    "text": self.prompt_func(tool, annotation_type)
                    }
                ],
            }
            ],
            )
        return message.content[0].text

class GrokExperiment(VisionExperiment):
    def __init__(self, model, prompt_func):
        super().__init__(model, prompt_func)
        self.client = OpenAI(api_key=os.getenv('GROK'), base_url="https://api.x.ai/v1")
    def process_sample(self, img_path, tool, annotation_type):
        converted_img_data, _ = self.convert_to_base64(img_path=img_path)
        messages = [
        {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{converted_img_data}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": self.prompt_func(tool, annotation_type),
            },
                    ],
                },
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.01,
        )
        return completion.choices[0].message.content
        