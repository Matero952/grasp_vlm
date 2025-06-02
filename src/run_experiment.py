from agents.claude_grok import ClaudeExperiment, GrokExperiment
from agents.gemini import GeminiExperiment
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
def run_owl(experiment, ground_truth_csv, iou_tolerance = None):
    os.makedirs("results", exist_ok=True)
    save_dir = os.path.join("results", 'owlvit-base-patch32')
    os.makedirs(save_dir, exist_ok=True)
    new_df_path = os.path.join(save_dir, f"owlvit-base-patch32.csv")
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path, sep=';', encoding='utf-8')
    else:
        df = pd.DataFrame(columns=["img_id", "img_path", "pred_bbox", "noun", "target_bbox", "iou"])
    with open(ground_truth_csv) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if str(row['img_id']) in df['img_id'].astype(str).values:
                print(f"Skipping: {str(row['img_id'])}")
            else:
                img = Image.open(str(row['img_path']))
                tensor_img = T.ToTensor()(img) 
                print(f"TENSOR IMG SHAPE: {tensor_img.shape}") 
                #tried using pillow but it looks like max set it up for tensors so converted to img tensor
                if 'index' in str(row['annotation_type']):
                    noun = ['trigger']
                else:
                    noun = ['handle']
                bboxs, _= experiment.predict(tensor_img, noun)
                print(bboxs)
                breakpoint()
                best_box = bboxs[noun[0]]['boxes'][torch.argmax(bboxs[noun[0]]['scores'])]
                print(best_box)
                print(ast.literal_eval(row['bbox']))
                breakpoint()
                # converted_box = reformat_owl(best_box.tolist(), tensor_img, owl_shape_tensor)
                # print(converted_box)
                # print(best_box)
                # print(type(best_box))
                # best_box = best_box.tolist()
                # print(best_box)
                # print(ast.literal_eval(row['bbox']))
                # top_left_x = best_box[0]
                # top_left_y = best_box[1]
                # bot_right_x = best_box[2]
                # bot_right_y = best_box[3]
                # print(top_left_x, type(top_left_x))
                # print(top_left_y, type(top_left_y))
                # print(bot_right_x, type(bot_right_x))
                # print(bot_right_y, type(bot_right_y))
                best_box = best_box.tolist()

                iou = get_iou(best_box, ast.literal_eval(row['bbox']))
                print(iou)
                breakpoint()
                # df.loc[len(df)] = [row['img_id'], row['img_path'], sanitize_text(clean_text(str(best_box))), noun[0], row['bbox']]
                # reformatted = reformat_owl(best_box)
                #not going to reformat because ground_truth_owl.csv is already in owls format

                # df.loc[len(df)] = [sanitize_text(clean_text(row['img_id'])), sanitize_text(clean_text(row['img_path'])), sanitize_text(clean_text(str()))]




def run_experiment(experiment, ground_truth_csv, iou_tolerance = None):
    os.makedirs("results", exist_ok=True)
    save_dir = os.path.join("results", experiment.model)
    os.makedirs(save_dir, exist_ok=True)
    new_df_path = os.path.join(save_dir, f"{experiment.model}_reasoning.csv")
    if iou_tolerance is not None:
        correct = 0
        seen = 0
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path, sep=';')
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
                df.loc[len(df)] = [row['img_id'], row['img_path'], sanitize_text(clean_text(text)), pred_bbox, target_bbox, iou]
                if iou_tolerance is not None and pred_bbox is not None:
                    correct += 1 if iou > iou_tolerance else 0
                    seen += 1
                df.to_csv(new_df_path, index=False, sep=';')
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
if __name__ == "__main__":
    # run_experiment(GrokExperiment("grok-2-vision-1212", generate_prompt), "src/ground_truth.csv")
    # run_experiment(GeminiExperiment("gemini-2.0-flash-lite", generate_prompt), "src/ground_truth.csv")
    run_owl(OWLv2(), 'src/ground_truth_owl.csv')



    