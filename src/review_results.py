#Claude and grok didnt return their results in the format that I wanted, so this is function is
#just for going back into their csvs and finding their bounding box answer.
import csv
import regex as re
import ast
import pandas as pd
from run_experiment import get_iou, get_pred_bbox_area
def review_results(input_csv_file, ground_truth_file):
    #claude results
    with open(input_csv_file) as f:
        reader = csv.DictReader(f)
        rows = []
        df = pd.read_csv(ground_truth_file)
        for row in reader:
            # print(row.keys())
            #csv field names have leading white spaces so its either 'iou' or ' iou'
            if row['iou'] == 'No predicted bbox found.':
                numb_match = re.findall(r'\d+\.?\d*', row['text_output'])
                if numb_match:
                    to_check_row = df[df['img_id'] == row['img_id']]
                    pred_bbox_coords = [int(float(i)) for i in numb_match[-4:]]
                    target_bbox = ast.literal_eval(row['target_bbox'])
                    iou = get_iou(pred_bbox_coords, target_bbox)
                    row['pred_bbox'] = pred_bbox_coords
                    row['iou'] = iou
                    #first, we dont really care about denormalizing lol
            rows.append(row)
        rows2 = []
        #now we check if we need to denormalize
        for row in rows:
            if sum(int(float(i)) for i in ast.literal_eval(row['pred_bbox'])) == 0:
                numb_match = numb_match = re.findall(r'\d+\.?\d*', row['text_output'])
                if numb_match:
                    to_check_row = df[df['img_id'] == row['img_id']]
                    pred_bbox_coords = denormalize([float(i) for i in numb_match[-4:]], 
                                        ast.literal_eval(to_check_row['image_dim'].values[0]))
            else:
                pred_bbox_coords = ast.literal_eval(row['pred_bbox'])
            target_bbox = ast.literal_eval(row['target_bbox'])
            iou = get_iou(pred_bbox_coords, target_bbox)
            row['pred_bbox'] = pred_bbox_coords
            row['iou'] = iou
            rows2.append(row)
        with open(input_csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows2)
        return rows2


    #         if sum([int(float(i)) for i in ast.literal_eval(row['pred_bbox'])]) == 0 or (row['iou'] == 'No predicted bbox found.'):
    #         # if row['iou'] == "No predicted bbox found.":
    #             numb_match = re.findall(r'\d+\.?\d*', row['text_output'])
    #             if numb_match and sum([float(i) for i in numb_match[-4:]]) == 0:
    #                 to_check_row = df[df['img_id'] == row['img_id']]
    #                 pred_bbox_coords = denormalize([float(i) for i in numb_match[-4:]], 
    #                                                    ast.literal_eval(to_check_row['image_dim'].values[0]))
    #                 # if sum([int(float(i)) for i in numb_match[-4:]]) == 0:
    #                 #     print(True)
    #                 #     to_check_row = df[df['img_id'] == row['img_id']]
    #                 #     pred_bbox_coords = denormalize([float(i) for i in numb_match[-4:]], 
    #                 #                                    ast.literal_eval(to_check_row['image_dim']))
    #             else:
    #                 pred_bbox_coords = [int(float(i)) for i in numb_match[-4:]]
    #                 target_bbox = ast.literal_eval(row['target_bbox'])
    #                 iou = get_iou(pred_bbox_coords, target_bbox)
    #                 row['img_id'] = row['img_id']
    #                 row['img_path'] = row['img_path']
    #                 row['text_output'] = row['text_output']
    #                 row['pred_bbox'] = pred_bbox_coords
    #                 row['target_bbox'] = row['target_bbox']
    #                 row['iou'] = iou
    #         rows.append(row)
            
    # # print(rows)
    # with open(input_csv_file, 'w', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
    #     writer.writeheader()
    #     writer.writerows(rows)
    # return rows
                    

def denormalize(normalized_bnd_box, img_dim):
    #claude normalized its results, so i have to denormalize them to get good data.
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalized_bnd_box
    width, height = img_dim
    return [x_min_norm * width, y_min_norm * height, x_max_norm * width, y_max_norm * height]
                       
review_results('results/claude-3-5-haiku-latest/claude-3-5-haiku-latest_results_w_reasoning.csv', 'src/ground_truth.csv')