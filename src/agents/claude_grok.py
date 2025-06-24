import anthropic
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import mimetypes
import re
load_dotenv()

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
    def process_sample(self, file_path, tool, vlm_role, model, task, bboxes: dict):
        converted_img_data, mime_type = self.convert_to_base64(img_path=file_path)
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
                    "text": self.prompt_func(file_path, tool, vlm_role, model, task, bboxes)
                    }
                ],
            }
            ],
            )
        return message, message.content[0].text

class GrokExperiment(VisionExperiment):
    def __init__(self, model, prompt_func):
        super().__init__(model, prompt_func)
        self.client = OpenAI(api_key=os.getenv('GROK'), base_url="https://api.x.ai/v1")
    def process_sample(self, file_path, tool, vlm_role, model, task, bboxes: dict):
        converted_img_data, _ = self.convert_to_base64(img_path=file_path)
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
                "text": self.prompt_func(file_path, tool, vlm_role, model, task, bboxes),
            },
                    ],
                },
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=0.01,
        )
        return completion, completion.choices[0].message.content
    
class GPTExperiment(VisionExperiment):
    def __init__(self, model, prompt_func):
        super().__init__(model, prompt_func)
        self.client = OpenAI(api_key=os.getenv('GPT'))
    def process_sample(self, file_path, tool, vlm_role, model, task, bboxes):
        converted_img_data, mime_type = self.convert_to_base64(img_path=file_path)
        response = self.client.responses.create(
            model=self.model,
            input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": self.prompt_func(file_path, tool, vlm_role, model, task, bboxes) },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{converted_img_data}",
                            },
                        ],
                    }
                ],
        )
        return response, response.output_text
if __name__ == "__main__":
    from src.prompt import get_prompt
    import pandas as pd
    import ast
    gem_exp = ClaudeExperiment('claude-3-5-haiku-latest', get_prompt)
    df = pd.read_csv('ground_truth_test.csv', sep=';')
    row = df.iloc[0]
    print(row)
    file_path = row['img_path']
    tool = row['tool']
    vlm_role = row['vlm_role']
    model = gem_exp.model
    task = row['task']
    bboxes = ast.literal_eval(row['bboxes'])
    response, text = gem_exp.process_sample(file_path, tool, vlm_role, model, task, bboxes)
    print(response)
    print(text)
    print(vars(response))
    # print(response.usage.input_tokens)
    # print(response.usage.output_tokens)
    #claude

    # print(response.usage.prompt_tokens)
    # print(response.usage.completion_tokens)
    #grok

    # input_tokens = response.usage.input_tokens
    # output_tokens = response.usage.output_tokens
    #gpt
    # print(input_tokens, output_tokens)
        
