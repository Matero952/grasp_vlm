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

def run_experiment(experiment, ground_truth_csv, iou_tolerance = None):
    os.makedirs("results", exist_ok=True)
    save_dir = os.path.join("results", experiment.model)
    os.makedirs(save_dir, exist_ok=True)
    new_df_path = os.path.join(save_dir, f"{experiment.model}_results.csv")
    if iou_tolerance is not None:
        correct = 0
        seen = 0
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path)
    else:
        df = pd.DataFrame(columns=["img_id", "img_path", "text_output", "pred_bbox", "target_bbox", "iou"])
    with open(ground_truth_csv) as f:
        reader = csv.DictReader(f)
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
                df.loc[len(df)] = [row['img_id'], row['img_path'], text, pred_bbox, target_bbox, iou]
                if iou_tolerance is not None and pred_bbox is not None:
                    correct += 1 if iou > iou_tolerance else 0
                    seen += 1
                df.to_csv(new_df_path, index=False)
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
if __name__ == "__main__":
    run_experiment(ClaudeExperiment("claude-3-haiku-20240307", generate_prompt), "src/ground_truth.csv")



    