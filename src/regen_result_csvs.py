import csv
import regex as re
#these functions are used to clean up csvs as on the first experiment they came out a little funky
from extract_reform import get_bnd_boxes, needs_denormalize, denormalize
from run_experiment import get_iou, get_pred_bbox_area
import pandas as pd
import ast
import os

#reform gemini will work with gemini responses but not with claude or grok
def reform(txt_file, ground_truth='src/ground_truth.csv'):
    #will work for the messed up gemini files, will not work for anything else.
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
    if 'gemini' in txt_file:
        # responses = split_by_custom_delim(txt_file, r"'''\"\n")
        responses = split_by_indentation(txt_file)
    elif 'claude' in txt_file:
        # responses = split_by_custom_delim(txt_file, r'"""\n|I apologize')
        responses = split_by_custom_delim(txt_file, r'"\n')
    elif 'grok' in txt_file:
        # responses = split_by_custom_delim(txt_file, r'To determine')
        responses = split_by_indentation(txt_file)
    print(f"len responses: {len(responses)}")
    print(f'0: {responses[0]}')
    print(f"-1: {responses[-1]}")
    # del(responses[-1])
    #here, the last one was just blank for my delimiter search
    pred_bnd_boxes = get_bnd_boxes(responses)
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
    return cleaned.strip()

def check(csv):
    df = pd.read_csv(csv, sep=';')
    print(df.columns)
    breakpoint()
    print(df['text_output'])
    print(len(df['text_output']))

def pls_grok(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        counter = 0
        for line in lines:
            if 'To determine' in line:
                counter += 1
    print(counter)
# print(len(get_bnd_boxes(split_by_indentation('results/raw_text/grok-2-vision-1212/grok-2-vision-1212.txt'))))
reform('results/raw_text/grok-2-vision-1212-reasoning/grok-2-vision-1212-reasoning.txt')
# pls_grok('results/raw_text/grok-2-vision-1212-reasoning/grok-2-vision-1212-reasoning.txt')
# split_by_custom_delim('results/raw_text/claude-3-5-haiku-latest-reasoning/claude-3-5-haiku-latest-reasoning.txt', r'"""\n|I apologize')
# check('output.csv')
#lets have a csv checking pipeline where we check if the csv is first of all ok, if it isnt, we refer back to the text file to fix it basically.