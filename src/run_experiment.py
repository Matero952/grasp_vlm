from agents import *
from prompt import *
import os
import pandas as pd
def run_experiment(model, prompt_function):
    os.makedirs("results", exist_ok=True)
    save_dir = os.path.join("results", model)
    os.makedirs(save_dir, exist_ok=True)
    new_df_path = os.path.join(save_dir, f"{model}_results.csv")
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path)
    else:
        df = pd.DataFrame(columns=["img_id", "text_output", "pred_bbox"])