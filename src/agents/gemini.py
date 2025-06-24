import google.genai as genai
import os
import regex
#This is the class for the Gemini vision testing
class GeminiExperiment:
    def __init__(self, model, prompt_func):
        self.model = model
        self.prompt_func = prompt_func
        self.client = genai.Client(api_key=os.environ.get("GEMINI"))

    def process_sample(self, file_path, tool, vlm_role, model, task, bboxes: dict):
        img = self.client.files.upload(file=file_path)
        response = self.client.models.generate_content(
            model=self.model,
            contents = [img, self.prompt_func(file_path, tool, vlm_role, model, task, bboxes)]
        )
        return response, response.text
    
if __name__ == '__main__':
    from src.prompt import get_prompt
    import pandas as pd
    import ast
    # from src.run_experiment import get_pred_bbox
    gem_exp = GeminiExperiment('gemini-2.5-flash-lite-preview-06-17', get_prompt)
    df = pd.read_csv('ground_truth_test.csv', sep=';')
    row = df.iloc[2]
    print(row)
    file_path = row['img_path']
    tool = row['tool']
    vlm_role = row['vlm_role']
    model = gem_exp.model
    task = row['task']
    bboxes = ast.literal_eval(row['bboxes'])
    response, text = gem_exp.process_sample(file_path, tool, vlm_role, model, task, bboxes)
    print(text)
    bboxes = regex.search(r'\{(?:[^{}]|(?R))*\}', text)
    print(bboxes.group(0))
    print(ast.literal_eval(bboxes.group(0)))
    for key, item in ast.literal_eval(bboxes.group(0)).items():
        print(type(item))
    # numbers_match = re.findall(r'\b\d+\.\d+|\b\d+|\B\.\d+', text)
    # bbox = [float(i) for i in numbers_match[-4:]]
    # print(bbox)
    # pred_bbox = get_pred_bbox(text, int(row['img_id']))
    # print(pred_bbox)
    breakpoint()
    print(response)
    print(text)
    print(vars(response))
    print(response.usage_metadata)
    print("Prompt tokens:", response.usage_metadata.prompt_token_count)
    print("Output tokens:", response.usage_metadata.candidates_token_count)
    print("Total tokens:", response.usage_metadata.total_token_count)
# /

#     print(f"Input tokens: {input_tokens}")
#     print(f"Output tokens: {output_tokens}")
#     print(f"Total tokens: {total_tokens}")

    
