#this file is meant to contain some functions if csv data gets COMPLETELY messed up.
#during testing, some of the claude data got completely messed up, so these methods were used to recover key information
#from that experiment. Thus, these functions are not readily applicable to other experiments, but can be easily tweaked.
#Use reform_bad_columns when: all of the columns are messed up. Otherwise, use reform.
import regex as re
import pandas as pd
import ast
from run_experiment import get_iou, get_pred_bbox_area
import csv
def reform_bad_columns(output_csv_path, txt_file, ground_truth_file='src/ground_truth.csv'):
    #aggregates all of the saved data and makes it into a new df.
    #saves to a csv output_csv_path
    #only use this if you need to completely reform the csv because the data is messed up.
    #not readily available to just use, but minimal tweaking required
    indv_responses = organize_responses(txt_file)
    print(len(indv_responses))
    bnd_boxes = get_bnd_boxes(indv_responses)
    print(len(bnd_boxes))
    print(bnd_boxes)
    df = pd.read_csv(ground_truth_file)
    rows = []
    for i in range(0, 200):
        to_check_row = df[df['img_id'].astype(str) == str(i)]
        img_id = to_check_row['img_id'].values[0]
        img_path = to_check_row['img_path'].values[0]
        text_output = indv_responses[i]
        pred_bbox = bnd_boxes[i]
        if len(pred_bbox) != 4:
            pred_bbox = [0, 0, 0, 0]
        print(f"pred_bbox:  {pred_bbox}")
        print(text_output)
        try:
            # print(pred_bbox)
            # print(type(pred_bbox))
            # for i in pred_bbox:
            #     print(type(i))
            # # breakpoint()
            pred_bbox_area = get_pred_bbox_area(pred_bbox)
            print(pred_bbox_area)
        except ValueError:
            print(f"pred_bbox: {pred_bbox}")
            print(type(pred_bbox))
            for i in pred_bbox:
                print(type(i))
            breakpoint()
        target_bbox = ast.literal_eval((to_check_row['bbox'].astype(str)).iloc[0])
        print("HI")
        print(target_bbox)
        print(type(target_bbox))
        for i in target_bbox:
            print(type(i))
        target_bbox = [float(i) for i in target_bbox]
        print("HIII")
        iou = get_iou(target_bbox, pred_bbox)
        print("HIIIIII")
        print(img_id)
        print(img_path)
        print(text_output)
        print(pred_bbox)
        print(pred_bbox_area)
        print(target_bbox)
        print(iou)
        row = [img_id, img_path, text_output, str(pred_bbox), pred_bbox_area, str(target_bbox), iou]
        rows.append(row)
    new_df = pd.DataFrame(rows, columns=['img_id', 'img_path', 'text_output', 'pred_bbox', 'pred_bbox_area', 'target_bbox', 'iou'])
    new_df.to_csv(output_csv_path)
    return new_df

def reform(input_csv_path, output_csv_path, ground_truth_file='src/ground_truth.csv'):
    result_df = pd.read_csv(input_csv_path)
    gt_df = pd.read_csv(ground_truth_file)
    new_df = pd.DataFrame(columns=['img_id', 'img_path', 'text_output', 'pred_bbox', 'pred_bbox_area', 'target_bbox', 'iou'])
    indv_responses = result_df['text_output'].to_list()
    bnd_boxes = get_bnd_boxes(indv_responses)
    rows = []
    for i in range(0, 200):
        to_check_row_gt = gt_df[gt_df['img_id'].astype(str) == str(i)]
        to_check_row_result = result_df[result_df['img_id'].astype(str) == str(i)]
        img_id = i
        img_path = to_check_row_gt['img_path']
        text_output = indv_responses[i]
        pred_bnd_box = bnd_boxes[i]
        pred_bnd_box_area = get_pred_bbox_area(pred_bnd_box)
        target_bbox = ast.literal_eval((to_check_row_gt['bbox'].astype(str)).iloc[0])
        target_bbox = [float(i) for i in target_bbox]
        iou = get_iou(target_bbox, pred_bnd_box)
        row = [img_id, img_path, text_output, pred_bnd_box, pred_bnd_box_area, target_bbox, iou]
        rows.append
    new_df.to_csv(output_csv_path)
    return new_df


def organize_responses(txt_file:str) -> list: 
    #this is a text file of ONLY the models text outputs, nothing else
    with open(txt_file, 'r', encoding='utf-8') as f:
        next(f)
        next(f)
        #SKIP AS MANY LINES AS YOU NEED TO MAKE THE EXTRACTION WORK
        #skipping the first line because it doesnt 
        #have a 'Let' so it messes up further extraction
        content = f.read()
    # responses = re.split(r'(?i)(?=Let)', content)
    responses = re.split(r'"\\n"', repr(content))
    indv_responses = [i.strip() for i in responses if i.strip()]
    # print(len(indv_responses))
    #this needs to be 200
    # breakpoint()
    return indv_responses

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
    df = pd.read_csv(gt_file)
    to_check_row = df[df['img_id'].astype(str) == str(idx)]
    image_dims = ast.literal_eval(to_check_row['image_dim'].iloc[0])
    width, height = [float(i) for i in image_dims]
    x_min, y_min, x_max, y_max = normalized_bnd_box
    return [x_min * width, y_min * height, x_max * width, y_max * height]
# def clean_text(text):
#     if not isinstance(text, str):
#         return text
#     text = re.sub(r'\s+', ' ', text)           
#     return text.strip()   

# def compact(csv_file, output_csv_path):
#     df = pd.read_csv(csv_file)
    
#     # Apply cleaning only to the 'text_output' column (or all object columns)
#     text_columns = df.select_dtypes(include='object').columns
#     for col in text_columns:
#         df[col] = df[col].apply(clean_text)
#     df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
            
if __name__ == "__main__":
    # reform_bad_columns('results/claude-3-5-haiku-latest/claude-3-5-haiku-latest_reformed.csv', 'results/raw_text_outputs/claude-3-5-haiku-latest_results_w_reasoning.txt')
    reform_bad_columns('results/gemini-2.0-flash/gemini-2.0-flash_results_w_reasoning_reformed.csv', 'results/raw_text_outputs/gemini-2.0-flash_w_reasoning.txt')
    # with open('results/raw_text_outputs/claude-3-haiku-20240307_results_w_reasoning.txt', 'r', encoding='utf-8') as f:
    #     content = f.read()
    #     print(repr(content))
    # compact('results/gemini-2.0-flash/gemini-2.0-flash_results_w_reasoning_reformed.csv', 'src/test.csv')
