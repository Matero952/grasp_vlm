import csv
import regex as re
#these functions are used to clean up csvs as on the first experiment they came out a little funky
from run_experiment import get_iou, get_pred_bbox_area
import pandas as pd
import ast
import os
#TODO NEED TO UPDATE GEMINI RESULTS
#reform gemini will work with gemini responses but not with claude or grok
def reform(txt_file, ground_truth='src/ground_truth.csv'):
    #will work for the messed up gemini files, will not work for anything else.
    output_csv = os.path.basename(txt_file)
    output_csv = os.path.splitext(output_csv)[0]
    if 'reasoning' in txt_file:
        output_csv = f"{output_csv}.csv"
    else:
        output_csv = f"{output_csv}.csv"
    rows = []
    df = pd.read_csv(ground_truth, sep=';')
    df.columns = df.columns.str.replace('"', '', regex=False)
    print(df.columns)
    if 'gemini' in txt_file:
        # responses = split_by_custom_delim(txt_file, r"'''\"\n")
        responses = split_by_indentation(txt_file)
    elif 'claude' in txt_file:
        # responses = split_by_custom_delim(txt_file, r'"""\n|I apologize')
        responses = split_by_custom_delim(txt_file, r'"\n')
    elif 'grok' in txt_file:
        # responses = split_by_custom_delim(txt_file, r'To determine')
        responses = split_by_indentation(txt_file)
    else:
        responses = split_by_indentation(txt_file)
    print(f"len responses: {len(responses)}")
    print(f'0: {responses[0]}')
    print(f"-1: {responses[-1]}")
    # del(responses[-1])
    #here, the last one was just blank for my delimiter search

    pred_bnd_boxes = get_bnd_boxes(responses)
    if 'gemini' in txt_file:
        for idx, i  in enumerate(pred_bnd_boxes):
            gt_col = df[df['img_id'].astype(str) == str(idx)]
            width, height = ast.literal_eval(gt_col['image_dim'].iloc[0])
            try:
                pred_bnd_boxes[idx] = format_gemini(i, width, height)
            except ValueError:
                pred_bnd_boxes[idx] = [0, 0, 0, 0]
    print(f"len bound box: {len(pred_bnd_boxes)}")
    # df.columns = df.columns.str.replace('"', '', regex=False)
    #universally cleans up df columns
    target_bnd_boxes = [ast.literal_eval(bbox) for bbox in df['bbox']]
    print(len(responses))
    print(len(pred_bnd_boxes))
    # print(target_bnd_boxes)
    print(len(target_bnd_boxes))
    for i in range(0, 200):
        print(len(responses))
        print(len(pred_bnd_boxes))
        row = {'img_id' : i, 'text_output' : clean_text(sanitize_text(responses[i])), 'pred_bbox' : pred_bnd_boxes[i], 'target_bbox' : target_bnd_boxes[i], 'iou' : get_iou(pred_bnd_boxes[i], target_bnd_boxes[i])}
        rows.append(row)
        print(repr(responses[i]))
    new_df = pd.DataFrame(rows)
    # new_df.to_csv(output_csv, sep=';', encoding='utf-8', index=False)
    new_df.to_csv(output_csv, sep=';', encoding='utf-8', index=False)
    return None

def reform_claude(txt_file, ground_truth='src/ground_truth.csv'):
    output_csv = os.path.basename(txt_file)
    output_csv = os.path.splitext(output_csv)[0]
    if 'reasoning' in txt_file:
        output_csv = f"{output_csv}_reason.csv"
    else:
        output_csv = f"{output_csv}.csv"
    rows = []
    df = pd.read_csv(ground_truth, sep=';')
    df.columns = df.columns.str.replace('"', '', regex=False)
    print(df.columns)

def split_by_custom_delim(filename, custom_delim):
    #this way of splitting text should work for the messed up claude files
    responses = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        clean_text(content)
        sanitize_text(content)
        responses = re.split(custom_delim, content)
    # for i in range(0, 200):
    #     print(responses[i])
    #     print(i)
    #     breakpoint()
    return responses
def split_by_indentation(filename):
    #splits and organizes responses based on their indentation in a txt file of the model responses.
    #works for all messed up gemini models
    responses = []
    current_block = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = sanitize_text(clean_text(line))
            indent = len(line) - len(line.lstrip('\t '))
            if indent == 0 and current_block:
                responses.append(''.join(current_block).rstrip())
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            # current_block = re.sub(r'\s+', ' ', str(current_block))
            responses.append(''.join(current_block).rstrip())
    print(len(responses))
    return responses[1:]

def sanitize_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace(';', '')
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
    cleaned = text.replace(';', '')
    return cleaned.strip()

def check(csv):
    df = pd.read_csv(csv, sep=';')
    print(df.columns)
    breakpoint()
    print(df['text_output'])
    print(len(df['text_output']))

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

def format_gemini(output_bnd_box, og_img_width, og_img_height) -> list:
    y_min, x_min, y_max, x_max = output_bnd_box
    y_min = y_min / 1000
    x_min = x_min / 1000
    y_max = y_max / 1000
    x_max = x_max / 1000
    return [int(x_min * og_img_width), int(y_min * og_img_height), int(x_max * og_img_width), int(y_max * og_img_height)]


def pls_grok(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        counter = 0
        for line in lines:
            if 'To determine' in line:
                counter += 1
    print(counter)

# print(len(get_bnd_boxes(split_by_indentation('results/raw_text/grok-2-vision-1212/grok-2-vision-1212.txt'))))
reform('results/raw_text/o4-mini_reasoning/o4-mini_reasoning.txt')
# pls_grok('results/raw_text/grok-2-vision-1212-reasoning/grok-2-vision-1212-reasoning.txt')
# split_by_custom_delim('results/raw_text/claude-3-5-haiku-latest-reasoning/claude-3-5-haiku-latest-reasoning.txt', r'"""\n|I apologize')
# check('output.csv')
#lets have a csv checking pipeline where we check if the csv is first of all ok, if it isnt, we refer back to the text file to fix it basically.