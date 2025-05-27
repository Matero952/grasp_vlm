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
        for row in reader:
        # for i, row in enumerate(reader, 1):
        #     if None in row:
        #         print(f"Row {i} has extra column(s):")
        #         # print(f"row: {row}")
        #         row = {
        #             k.strip() if isinstance(k, str) else k:
        #             v.strip() if isinstance(v, str) else v
        #             for k, v in row.items()
        #         }
        #         print(row)
        # breakpoint()




        # reader = csv.DictReader(f)
        # rows = []
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # df = pd.read_csv(ground_truth_file)
        # print(df['img_id'].values)
        # # for i in df['img_id'].values:
        # #     print(type(i))
        # #     breakpoint()
        # # breakpoint()
        # counter = 0
        # for row in reader:
        #     print(counter)
        #     counter += 1
        #     print(row.keys())
        #     if None in row.keys():
        #         print(f"row: {row}")
        #         print(f"row iou: {row[' iou']}")
        #         print(f"row none: {row[None]}")
        #         breakpoint()
        #     row = {key.strip(): value for key, value in row.items()}
        #     # print(row.keys())
        #     #csv field names have leading white spaces so its either 'iou' or ' iou'
        #     # print(row.keys())
        #     # if None in row.keys():
        #     #     breakpoint()
        #     # print(repr(row['iou']))
            if row['iou'] == 'No predicted bbox found.':
                numb_match = re.findall(r'\d+\.?\d*', row['text_output'])
                if numb_match:
                    # to_check_row = df[df['img_id'] == row['img_id']]
                    pred_bbox_coords = [int(float(i)) for i in numb_match[-4:]]
                    target_bbox = ast.literal_eval(row['target_bbox'])
                    iou = get_iou(pred_bbox_coords, target_bbox)
                    row['img_id'] = row['img_id']
                    row['img_path'] = row['img_path']
                    row['text_output'] = row['text_output']
                    row['pred_bbox'] = pred_bbox_coords
                    row['target_bbox'] = row['target_bbox']
                    row['iou'] = iou
                    #first, we dont really care about denormalizing lol
            rows.append(row)
        print(rows)
        breakpoint()
        rows2 = []
        #now we check if we need to denormalize
        counter = 0
        for row in rows:
            row = {key.strip(): value for key, value in row.items()}
            print(repr(row['pred_bbox']))
            print(type(row['pred_bbox']))
            for i in row['pred_bbox']:
                print(i)
                print(type(i))
            print(counter)
            foo_ls = [int(float(i)) for i in ast.literal_eval(row['pred_bbox'])]
            counter += 1
            print(foo_ls)
            print(type(foo_ls))
            if sum(foo_ls) == 0:
            # if sum([int(float(i)) for i in ast.literal_eval(row['pred_bbox'])]) == 0:
                numb_match = numb_match = re.findall(r'\d+\.?\d*', row['text_output'])
                if numb_match:
                    # print(df['img_id'].)
                    to_check_row = df[df['img_id'].astype(str) == str(row['img_id']).strip()]
                    #df['img_id'] is an array of numpy int 64s and row['img_id'] is an array of strings
                    #so we need to cast it
                    pred_bbox_coords = denormalize([float(i) for i in numb_match[-4:]], 
                                        ast.literal_eval((to_check_row['image_dim']).iloc[0]))
            else:
                pred_bbox_coords = ast.literal_eval(row['pred_bbox'])
            target_bbox = ast.literal_eval(row['target_bbox'])
            iou = get_iou(pred_bbox_coords, target_bbox)
            row['pred_bbox'] = pred_bbox_coords
            row['iou'] = iou
            rows2.append(row)
        print(rows2)
        breakpoint()
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