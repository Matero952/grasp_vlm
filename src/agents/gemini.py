import google.genai as genai
import os
import re
class GeminiExperiment:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt
        self.client = genai.Client(api_key=os.environ.get("GEMINI"))

    def process_sample(self, file_path):
        img = self.client.files.upload(file=file_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents = [img, self.prompt]
        )
        return response.text
