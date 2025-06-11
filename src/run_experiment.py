from agents.claude_grok import ClaudeExperiment, GrokExperiment
from agents.gemini import GeminiExperiment
from agents.claude_grok import GPTExperiment
from prompt import *
from prompt import generate_prompt
import os
import pandas as pd
import csv
import regex as re
import ast
import time
import numpy as np
from agents.owl import *
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np
from regen_result_csvs import *
# from graph import plot_box_and_whiskers, get_owl_single
from pathlib import Path
#Allen key ===> Allen wrench tool(allen wrench tool)
#Drill ===> Drill tool/power tool(tool)
#Glue gun ===> Glue gun(glue gun tool/tool)
#Hammer ===> Hammer construction tool(hammer/tool)
#Nail gun ===> Nail gun/nail gun construction tool(nail gun/tool)
#Wrench ===> Wrench/wrench tool(or just tool)
#Circular saw ===> Circular saw
tool_dict_for_yolo_1 = {'drill' : 'drill tool', 'wacker' : 'weed wacker', 'glue' : 'glue gun tool', 'saw' : 'circular saw', 'nail' : 'nail gun', 
    'screwdriver' : 'screwdriver', 'wrench' : 'wrench tool', 'solder' : 'solder iron tool', 'allen' : 'allen wrench tool', 'hammer' : 'hammer'}
tool_regex = r'drill|wacker|glue|saw|nail|screwdriver|wrench|solder|allen|hammer'

def generate_owl_prompts(base_directory):
    #generates owl prompts from old results
    owl_paths = []
    prompts = []
    
    path = Path(base_directory)
    for i in path.rglob('*.csv'):
        print(str(i))
        if 'owl' in str(i):
            print(str(i))
            owl_paths.append(str(i))
        else:
            continue
    assert len(owl_paths) == 13
    for i in owl_paths:
        assert isinstance(i, str)
        df = pd.read_csv(i, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False)
        to_check_row_hand = df.iloc[0]
        to_check_row_index = df.iloc[1]
        hand_prompt = to_check_row_hand['noun']
        index_prompt = to_check_row_index['noun']
        prompts.append((index_prompt, hand_prompt))
    print(prompts)
    return prompts

def run_owl_experiments(experiment, ground_truth_csv, prompt_base_dir = 'results'):
    prompts = generate_owl_prompts(prompt_base_dir)
    breakpoint()
    print(f"Starting OWL experiments for {prompts=}")
    counter = 0
    completed_experiment_list = []
    for i in prompts:
        os.makedirs('results', exist_ok=True)
        save_dir = os.path.join("results", 'owl')
        os.makedirs(save_dir, exist_ok=True)
        new_df_path = os.path.join(save_dir, f"owlvit-base-patch32_{counter}.csv")
        df = pd.DataFrame(columns=['img_id', 'img_path', 'pred_bbox', 'noun', 'target_bbox', 'iou'])
        df.columns = df.columns.str.replace('"', '', regex=False)
        with open(ground_truth_csv) as f:
            reader = csv.DictReader(f, delimiter=';')
            # reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                # print(row.keys())
                # breakpoint()
                # if 'img_path' in row.keys():
                #     print('AAAAAAAAA')
                # print(row['img_path'])
                # for col in df.columns:
                #     print(repr(list(col)))
                # print(row)
                img = Image.open(str(row['img_path']))
                img = T.ToTensor()(img)
                prompt_index, prompt_hand = i
                tool_match = re.search(tool_regex, row['img_path'])
                if tool_match:
                    tool = tool_match.group(0)
                else:
                    raise ValueError('No tool found in the image path.')
                if 'index' in str(row['annotation_type']):
                    prompt = f"{prompt_index} on the {tool_dict_for_yolo_1[tool]}"
                elif 'four' in str(row['annotation_type']):
                    prompt = f"{prompt_hand} on the {tool_dict_for_yolo_1[tool]}"
                else:
                    raise ValueError('No recognized annotation type')
                print(f'{prompt=}')
                prompt = [f'{prompt}']
                bboxs = experiment.predict(img, prompt)
                print(f'{bboxs=}')
                best_box = bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])]
                best_box = best_box.tolist()
                iou = get_iou(best_box, ast.literal_eval(row['bbox']))
                df.loc[int(row['img_id'])] = [row['img_id'], row['img_path'], str(best_box).strip(), str(prompt), row['bbox'], iou]
                df.to_csv(new_df_path, sep=';', encoding='utf-8', index=False)
        print(f"Finished OWL experiment for {i=}")
        print(f"Starting OWL experiment for {i=}")
        counter += 1
        completed_experiment_list.append(new_df_path)
    # plot_box_and_whiskers(get_owl_single(completed_experiment_list, 'src/ground_truth_owl.csv'))

def run_experiment(experiment, ground_truth_csv, delay, iou_tolerance = None, reasoning=False):
    os.makedirs("results", exist_ok=True)
    save_dir = os.path.join("results", experiment.model)
    os.makedirs(save_dir, exist_ok=True)
    if reasoning:
        new_df_path = os.path.join(save_dir, f"{experiment.model}_reasoning.csv")
    else:
        new_df_path = os.path.join(save_dir, f"{experiment.model}.csv")
    if iou_tolerance is not None:
        correct = 0
        seen = 0
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False)
    else:
        df = pd.DataFrame(columns=["img_id", "img_path", "text_output", "pred_bbox", "target_bbox", "iou"])
    with open(ground_truth_csv) as f:
        reader = csv.DictReader(f, delimiter=';')
        counter = 0
        for row in reader:
            if str(row['img_id']) in df['img_id'].astype(str).values:
                if iou_tolerance is not None:
                    to_check_row = df[df['img_id'] == row['img_id']]
                    correct += 1 if to_check_row['iou'].iloc[0] > iou_tolerance else 0
                    seen += 1
                print(f"Skipping: {row['img_id']}")
                continue
            else:
                text = experiment.process_sample(row['img_path'], row['tool'], row['annotation_type'])
                pred_bbox = get_pred_bbox(text)
                target_bbox = ast.literal_eval(row['bbox'])
                if pred_bbox is None:
                    iou = "No predicted bbox found."
                    pred_bbox_area = "No predicted bbox found."
                else:
                    iou = get_iou(pred_bbox, target_bbox)
                    pred_bbox_area = get_pred_bbox_area(pred_bbox)
                df.loc[len(df)] = [row['img_id'], row['img_path'], sanitize_text(clean_text(text)), sanitize_text(clean_text(pred_bbox)), sanitize_text(clean_text(target_bbox)), sanitize_text(clean_text(str(iou)))]
                if iou_tolerance is not None and pred_bbox is not None:
                    correct += 1 if iou > iou_tolerance else 0
                    seen += 1
                df.to_csv(new_df_path, index=False, sep=';')
                time.sleep(delay)
            if iou_tolerance is not None:
                print(f"img_id: {row['img_id']}, accuracy: {correct / seen}")
            else:
                print(f"img_id: {row['img_id']}")


#helper functions for extracting key information from responses
def get_pred_bbox(text: str):
    text = text.lower()
    pattern = r'[\[{(]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\]})]?'
    #searches for all instances of bound boxes in formats: (), [], or {}
    bbox_matches = re.findall(pattern, text)
    if bbox_matches:
        try:
            pred_bbox = ast.literal_eval(bbox_matches[-1])
        #last bound box is probably the final answer
        except SyntaxError:
            pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\D+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?){3}"
            pred_bbox_match = re.search(pattern, bbox_matches[-1])
    else:
        return None
    return list(pred_bbox)
    #returns it as a list

def get_iou(bbox_a, bbox_b):
    if len(bbox_a) < 4:
        bbox_a = [0, 0, 0, 0]
    if len(bbox_b) < 4:
        bbox_b = [0, 0, 0 ,0]
    x1_1, y1_1, x2_1, y2_1 = bbox_a
    x1_2, y1_2, x2_2, y2_2 = bbox_b    
    #get intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    #get intersection area
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection_area = 0
    else:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    #get union area
    union_area = area1 + area2 - intersection_area
    #ge IoU
    if union_area == 0:
        return 0.0
    iou = intersection_area / union_area
    return iou

def get_pred_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height
def sanitize_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('"', '')
    text = text.replace('""', '')
    text = text.replace('""""', '')  
    text = text.replace('\n', ' ').replace('\r', ' ')
    return f"{text.strip()}"

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Replace all runs of whitespace (spaces, newlines, tabs) with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def reformat_owl(box, from_size_tensor, owl_tensor):
    #converts (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    #to (bot_x1, bot_y1, top_x2, top_y2)'
    assert isinstance(from_size_tensor, torch.Tensor)
    assert isinstance(owl_tensor, torch.Size)
    x1, y1, x2, y2 = box
    _, og_height, og_width = from_size_tensor.shape
    print(owl_tensor)
    print(f"og height: {og_height}, og_width: {og_width}")
    breakpoint()
    _, _, owl_height, owl_width = owl_tensor
    print(f"owl height: {owl_height}, owl_width: {owl_width}")

    fx = owl_width / og_width
    print(fx)
    fy = owl_height / og_height
    print(fy)
    return [x1 * fx, y1 * fy, x2 * fx, y2 * fy]

def check_gpt(path):
    df = pd.read_csv(path, sep=';')
    print(df.columns)
    print(len(df.columns))
    print(df.iloc[0])
if __name__ == "__main__":
    # run_experiment(GrokExperiment("grok-2-vision-1212", generate_prompt), "src/ground_truth.csv")
    # run_experiment(GeminiExperiment("gemini-2.0-flash-lite", generate_prompt), "src/ground_truth.csv")
    # print(generate_owl_prompts('results'))
    # breakpoint()
    gem_models = [('gemini-2.0-flash', 5), ('gemini-2.0-flash-lite', 2.5), ('gemini-2.5-flash-preview-05-20', 5.5)]
    for i in gem_models:
        model, delay = i
        run_experiment(GeminiExperiment(model, generate_prompt), "src/ground_truth.csv", reasoning=False, delay=delay)
    # check_gpt('results/o4-mini/o4-mini_reasoning.csv')



    