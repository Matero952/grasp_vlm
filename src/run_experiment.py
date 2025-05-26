from agents.claude_grok import ClaudeExperiment, GrokExperiment
from agents.gemini import GeminiExperiment
from prompt import *
import os
import pandas as pd
import csv
def run_experiment(experiment, ground_truth_csv):
    os.makedirs("results", exist_ok=True)
    save_dir = os.path.join("results", experiment.model)
    os.makedirs(save_dir, exist_ok=True)
    new_df_path = os.path.join(save_dir, f"{experiment.model}_results.csv")
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path)
    else:
        df = pd.DataFrame(columns=["img_id", "img_path", "text_output", "pred_bbox", "target_bbox", "iou"])
    with open(ground_truth_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row['img_id'])
        
    # for img_id in ground_truth_csv["img_id"].values():
    #     if img_id in df["img_id"].values():
    #         continue
    #     else:
    #         text = experiment.process_sample()

if __name__ == "__main__":
    run_experiment(GeminiExperiment("gemini-2.0-flash", "test"), "src/ground_truth.csv")

    