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
        result = (response.text).lower()
        grasp = re.search(r"palmar pinch|power disk|precision disk|power sphere|precision sphere|sphere 3 finger|sphere 4 finger|tripod|inferior pincer|quadpod|lateral|extension type|palmar", result)
        return response.text, grasp
if __name__ == "__main__":
    # print(vars(genai))
    gemexperiment = GeminiExperiment(model="gemini-2.0-flash", prompt="Identify the best grasp for this object based on Feix's grasp taxonomy. For example, if you saw a piece of pencil lead, you would say: tip pinch")
    res, answer = gemexperiment.process_sample("data/tennis/tennis1.jpg")
    print(res)
    print(answer)