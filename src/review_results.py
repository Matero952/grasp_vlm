#Claude and grok didnt return their results in the format that I wanted, so this is function is
#just for going back into their csvs and finding their bounding box answer.
import csv
import regex as re
import ast
from run_experiment import get_iou, get_pred_bbox_area
def review_results(input_csv_file):
    with open(input_csv_file) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            #csv field names have leading white spaces so i gotta fix this rq
            print(row.keys())
            breakpoint()
            if row['iou'] == "No predicted bbox found.":
                numb_match = re.findall(r'\d+\.?\d*', row['text_output'])
                if numb_match:
                    pred_bbox_coords = [int(float(i)) for i in numb_match[-4:]]
                    target_bbox = ast.literal_eval(row['target_bbox'])
                    iou = get_iou(pred_bbox_coords, target_bbox)
                    row['img_id'] = row['img_id']
                    row['img_path'] = row['img_path']
                    row['text_output'] = row['text_output']
                    row['pred_bbox'] = pred_bbox_coords
                    row['target_bbox'] = row['target_bbox']
                    row['iou'] = iou
                rows.append(row)
            
    print(rows)
    with open(input_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
                    

                    
review_results('results/claude-3-5-haiku-latest/claude-3-5-haiku-latest_results_w_reasoning.csv')