#this file is meant to contain some functions if csv data gets COMPLETELY messed up.
#during testing, some of the claude data got completely messed up, so these methods were used to recover key information
#from that experiment. 
import regex as re
import pandas as pd
import ast
from run_experiment import get_iou, get_pred_bbox_area
import csv
# import rbql
# import subprocess
# def reform_bad_columns(output_csv_path, txt_file, ground_truth_file='src/ground_truth.csv'):
#     #aggregates all of the saved data and makes it into a new df.
#     #saves to a csv output_csv_path
#     indv_responses = organize_responses(txt_file)
#     bnd_boxes = get_bnd_boxes(indv_responses)
#     df = pd.read_csv(ground_truth_file)
#     rows = []
#     print(len(indv_responses))

#     for i in range(0, 200):
#         to_check_row = df[df['img_id'].astype(str) == str(i)]
#         img_id = to_check_row['img_id'].values[0]
#         img_path = to_check_row['img_path'].values[0]
#         text_output = indv_responses[i].replace('\n', '').replace('\r', '').strip()
#         pred_bbox = bnd_boxes[i]
#         if len(pred_bbox) != 4:
#             pred_bbox = [0, 0, 0, 0]
#         try:
#             pred_bbox_area = get_pred_bbox_area(pred_bbox)
#         except ValueError:
#             breakpoint()
#         target_bbox = ast.literal_eval((to_check_row['bbox'].astype(str)).iloc[0])
#         target_bbox = [float(i) for i in target_bbox]
#         iou = get_iou(target_bbox, pred_bbox)
#         row = [img_id, img_path, text_output, str(pred_bbox), pred_bbox_area, str(target_bbox), iou]
#         rows.append(row)
#     new_df = pd.DataFrame(rows, columns=['img_id', 'img_path', 'text_output', 'pred_bbox', 'pred_bbox_area', 'target_bbox', 'iou'])
#     new_df.to_csv(output_csv_path, sep=',', index=False, quoting=csv.QUOTE_ALL)
#     return new_df

# #reform is not really necessary reform_bad_columns kinda does everything
# # def reform(input_csv_path, output_csv_path, ground_truth_file='src/ground_truth.csv'):
# #     result_df = pd.read_csv(input_csv_path)
# #     gt_df = pd.read_csv(ground_truth_file)
# #     new_df = pd.DataFrame(columns=['img_id', 'img_path', 'text_output', 'pred_bbox', 'pred_bbox_area', 'target_bbox', 'iou'])
# #     indv_responses = result_df['text_output'].to_list()
# #     bnd_boxes = get_bnd_boxes(indv_responses)
# #     rows = []
# #     for i in range(0, 200):
# #         to_check_row_gt = gt_df[gt_df['img_id'].astype(str) == str(i)]
# #         to_check_row_result = result_df[result_df['img_id'].astype(str) == str(i)]
# #         img_id = i
# #         img_path = to_check_row_gt['img_path']
# #         text_output = indv_responses[i]
# #         pred_bnd_box = bnd_boxes[i]
# #         pred_bnd_box_area = get_pred_bbox_area(pred_bnd_box)
# #         target_bbox = ast.literal_eval((to_check_row_gt['bbox'].astype(str)).iloc[0])
# #         target_bbox = [float(i) for i in target_bbox]
# #         iou = get_iou(target_bbox, pred_bnd_box)
# #         row = [img_id, img_path, text_output, pred_bnd_box, pred_bnd_box_area, target_bbox, iou]
# #         rows.append
# #     new_df.to_csv(output_csv_path)
# #     return new_df


# def organize_responses(txt_file: str) -> list:
#     with open(txt_file, 'r', encoding='utf-8') as f:
#         next(f)  # Skip the first line if needed

#         # Read the rest of the file
#         content = f.read()

#     # Remove completely empty lines to avoid blank entries
#     cleaned_lines = [line for line in content.splitlines() if line.strip() != '']
#     cleaned_content = '\n'.join(cleaned_lines)

#     # Split responses on the delimiter your data uses
#     # Here I keep your split by '"\n"', which you used before
#     responses = cleaned_content.split('"\n"')

#     # Sanitize each response:
#     indv_responses = []
#     for r in responses:
#         # Strip extra quotes and whitespace around each response
#         sanitized = r.strip().strip('"')

#         # Replace newlines and carriage returns inside the response with spaces
#         sanitized = sanitized.replace('\n', ' ').replace('\r', ' ')

#         # Optionally collapse multiple spaces into one (cleaner)
#         sanitized = re.sub(r'\s+', ' ', sanitized)

#         indv_responses.append(str(sanitized).strip())

#     # Debug prints (optional)
#     print(f"Total responses after cleaning: {len(indv_responses)}")

#     return indv_responses

def get_bnd_boxes(indv_responses: list) -> list:
    bnd_boxes = []
    counter = 0
    for idx, i in enumerate(indv_responses):
        i = i.replace('""', '"')
        numbers_match = re.findall(r'\b\d+\.\d+|\b\d+|\B\.\d+', i)
        if numbers_match:
            if needs_denormalize(numbers_match[-4:]):
                #runs check for denormalization because 
                #some bnd boxes are normalized and some are not
                bbox = denormalize(idx, numbers_match[-4:], 'src/ground_truth.csv')
                #denormalizing if the vlm outputted number normalized 0 - 1
            else:
                #otherwise we just take the vlm-outputted bnd box
                bbox = [float(i) for i in numbers_match[-4:]]
        else:
            bbox = [0, 0, 0, 0]
        bnd_boxes.append(bbox)
    return bnd_boxes

def needs_denormalize(raw_bnd_box) -> bool:
    #runs a check to see if denormalization is necessary
    foo = [float(i) for i in raw_bnd_box]
    for i in foo:
        if i <= 1.0:
            #checks if there is a float less than one,
            #which indicates that the bnd box has been normalized
            return True
        else:
            continue
    return False

def denormalize(idx, raw_bnd_box, gt_file) -> list:
    #indv responses are ordered correctly(they line up with img ids), so i can just get idx
    normalized_bnd_box = [float(i) for i in raw_bnd_box]
    df = pd.read_csv(gt_file, sep=';')
    df.columns = df.columns.str.replace('"', '', regex=False)
    to_check_row = df[df['img_id'].astype(str) == str(idx)]
    image_dims = ast.literal_eval(to_check_row['image_dim'].iloc[0])
    width, height = [float(i) for i in image_dims]
    x_min, y_min, x_max, y_max = normalized_bnd_box
    return [x_min * width, y_min * height, x_max * width, y_max * height]




    
if __name__ == "__main__":
    reform_bad_columns('results/gemini-2.0-flash/gemini-2.0-flash_results_test.csv', 'results/raw_text/gemini-2.0-flash/gemini-2.0-flash.txt')
    # check('results/claude-3-5-haiku-latest/claude-3-5-haiku-latest_w_reasoning_reformed.csv')
    # extract_responses('results/claude-3-haiku-20240307/claude-3-haiku-20240307_results_w_reasoning_reformed.csv')
    please_work()