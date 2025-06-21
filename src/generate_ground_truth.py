#file to generate ground truth csv for vlm testing
import json
import os
import regex as re
import pandas as pd
def generate_models_card(output_path):
    pass
def generate_ground_truth(rf_json_path, output_path):
    category_mapping = {}
    rows = []
    with open(rf_json_path, 'r') as f:
        dataset = json.load(f)
        for category_id in dataset['categories']:
            if category_id['id'] != 0:
                #Two category ids, '0' and '1' map to the same thing in the json
                category_mapping[category_id['id']] = category_id['name']
    for img_data in dataset['images']:
        print(img_data['width'])
        if int(img_data['width']) > 7000:
            print(f'{img_data=}')
            breakpoint()

if __name__ == "__main__":
    generate_ground_truth('grasp_vlm_dataset/_annotations.coco.json', "ground_truth.csv")
    