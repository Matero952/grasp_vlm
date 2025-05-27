import google.genai as genai
import os
import re
#This is the class for the Gemini vision testing
class GeminiExperiment:
    def __init__(self, model, prompt_func):
        self.model = model
        self.prompt_func = prompt_func
        self.client = genai.Client(api_key=os.environ.get("GEMINI"))

    def process_sample(self, file_path, tool, annotation_type):
        img = self.client.files.upload(file=file_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents = [img, self.prompt_func(tool, annotation_type)]
        )
        return response.text
