from agents.claude_grok import ClaudeExperiment, GrokExperiment
from agents.gemini import GeminiExperiment
from agents.claude_grok import GPTExperiment
from prompt import *
import os
import pandas as pd
import csv
import regex as re
import ast
import numpy as np
from agents.owl import *
from PIL import Image
import torchvision.transforms as T
import numpy as np
import json
from agents.claude_grok import VisionExperiment
from graph import *
def run_grasp_vlm_experiment(experiment_list:list[GeminiExperiment|VisionExperiment|OWLv2]):
    pth_list = []
    for experiment in experiment_list:
        save_pth = run_experiment(experiment)
        plot_prediction_grid(save_pth, 64)
        pth_list.append(save_pth)
    pth_list.append('results/yolo_uniow.csv')
    pth_list.append('results/yolo_world.csv')
    plot_numb_boxes_box_and_whiskers_comparison(pth_list)
    plot_box_and_whiskers_comparison(pth_list)
    plot_ious_by_img(pth_list)





def get_prompt_owl(ground_truth_row):
    row = ground_truth_row
    if len(ast.literal_eval(row['bboxes']).keys()) == 2:
        if row['tool'] in ['bowling ball', 'violin bow', 'syringe', 'dart', 'pair of scissors']:
            #index thumb finger tool
            prompt_1 = f"Placement for index finger on the {row['tool']}"
            prompt_2 = f"Placement for thumb finger on the {row['tool']}"
        else:
            #two handed tool
            prompt_1 = f"Where one entire hand can grab the {row['tool']} safely"
            prompt_2 = f"Where another entire hand can grab the {row['tool']} safely"
        return [prompt_1, prompt_2]
    else:
        if row['tool'] in ['allen key', 'hammer', 'screwdriver', 'wrench', 'soldering iron'] or 'handle' in row['tool']:
            #handle tools:
            prompt = f"Where an entire hand can grab the {row['tool']} safely"
        else:
            prompt = f"Where the index finger can press on the {row['tool']} safely"
        return [prompt]


def run_experiment(experiment: GeminiExperiment|VisionExperiment|OWLv2, ground_truth_csv_path='ground_truth_test.csv'):
    if isinstance(experiment, OWLv2):
        owl_experiment = True
        file_stem = 'owl_vit'
    else:
        owl_experiment = False
        file_stem = f'{experiment.model}' 
    os.makedirs('results', exist_ok=True)
    file_stem += '.csv'
    save_pth = os.path.join('results', file_stem)
    if os.path.exists(save_pth):
        df = pd.read_csv(save_pth, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False)
    else:
        if not owl_experiment:
            df = pd.DataFrame(columns=["img_id", "img_path", "text_output", "pred_bboxes", "target_bboxes", "ious", "input_tokens", "output_tokens"])
        else:
            df = pd.DataFrame(columns=["img_id", "img_path", "pred_bboxes", "target_bboxes", "ious", "prompts"])
    with open(ground_truth_csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if (df['img_path'].astype(str).str.strip().apply(get_file_check) == get_file_check(row['img_path'].strip())).any():
                #since we had to reexport the dataset, the img_id order and .rf extensions all changed, so the safest way to check if it has been done before is through the file name
                to_check_row = df[df['img_path'].astype(str).str.strip().apply(get_file_check) == get_file_check(row['img_path'].strip())]
                try:
                    to_check_row = to_check_row.iloc[0].to_dict()
                except IndexError:
                    assert 0 > 1
                print(f"continuing")
                continue
            else:
                if owl_experiment:
                    prompt = get_prompt_owl(row)
                    result = process_owl_output(experiment, row, prompt)
                    reformatted_bnd_boxes, result_iou_dict = calculate_iou_results(experiment, result, row)
                    df.loc[len(df)] = [row['img_id'], row['img_path'], reformatted_bnd_boxes, row['bboxes'], result_iou_dict, prompt]
                elif not owl_experiment:
                    print(experiment.model)
                    new_response, pred_dict, input_tokens, output_tokens = process_vlm_output(experiment, row)
                    try:
                        reformatted_bnd_boxes, result_iou_dict = calculate_iou_results(experiment, pred_dict, row)
                    except TypeError:
                        print(reformatted_bnd_boxes)
                        print(result_iou_dict)
                    df.loc[len(df)] = [row['img_id'], row['img_path'], new_response, reformatted_bnd_boxes, row['bboxes'], result_iou_dict, input_tokens, output_tokens]
                else:
                    assert 0 > 1, print('this should never happen?!')
                df.to_csv(f'results/{file_stem}', sep=';', encoding='utf-8', index=False)
    return save_pth


def process_vlm_output(experiment, row: dict):
    #running the experiment uses csv dictreaders, so make sure that the input to this is a row of the reader of type dictionary
    response, new_text  = experiment.process_sample(row['img_path'], row['tool'], row['vlm_role'], 
                                                                    experiment.model, row['task'], ast.literal_eval(row['bboxes']))
    new_text = new_text.replace('\n', '')
    new_text = re.sub(r'```json\s*', '```json', new_text)
    new_text = re.sub(r'\s*```$', '```', new_text)
    max_attempts = 3
    for attempt in range(max_attempts):
        pred_bboxes = re.search(r'\{(?:[^{}]|(?R))*\}', new_text)
        if pred_bboxes is not None:
            pred_bboxes = pred_bboxes.group(0)
            try:
                clean_str = str(pred_bboxes).replace("{{", "[{").replace("}}", "}]")
                pred_dict = json.loads(clean_str)
                assert isinstance(pred_dict, dict), (print('pred_dict needs to be a dict'), print(f'{type(pred_dict)=}'))
                break
            except ValueError:
                continue
    else:
        #now we handle where the model did not produce any viable json output
        target_keys = list(ast.literal_eval(row['bboxes']).keys())
        target_keys: list[str] = [i.strip() for i in target_keys]
        if len(target_keys) == 2:
            #this handles if there are two target bboxes that we need to fix
            if 'index' in target_keys:
                pred_dict =  {'index': [0, 0, 0, 0], 'thumb': [0, 0, 0, 0]}
            elif 'hand1' in target_keys:
                pred_dict =  {'hand1': [0, 0, 0, 0], 'hand2': [0, 0, 0, 0]}
            else:
                assert 0 > 1, (print(target_keys, print(row['bboxes']), print(f'If it is a two handed object its gotta be either index-thumb or hand1-hand2')))
        elif len(target_keys) == 1:
            if 'handle' in target_keys:
                pred_dict = {'hand': [0, 0, 0, 0]}
            elif 'index' in target_keys:
                pred_dict =  {'index': [0, 0, 0, 0]}
            else:
                assert 0 > 1, (print(target_keys), print(row['bboxes']), print('If it is a one handed object, then the annotaiton needs to be either hand or index'))
        else:
            assert 0 > 1, (print('target_keys is not an expected length'), print(target_keys), print(len(target_keys)), print(row['bboxes']))
    input_token_size, output_token_size = get_token_input_output_size(experiment, response)
    return new_text, pred_dict, input_token_size, output_token_size

def process_owl_output(experiment, row: dict, prompt):
    img = Image.open(row['img_path'])
    img = T.ToTensor()(img)
    bboxs = experiment.predict(img, prompt)
    print(bboxs)
    # breakpoint()
    #we dont need to keep on trying to get a response out of owl if it doesnt work because it is deterministic, if that img and prompt do not work, then they will never work.
    target_keys = list(ast.literal_eval(row['bboxes']).keys())
    target_keys: list[str] = [i.strip() for i in target_keys]
    if len(prompt) == 2:
        try:
            best_prompt_1  = bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])]
            best_prompt_1 = [float(i)/1000 for i in best_prompt_1]
            print(best_prompt_1)
            #the prompts if its an index-thumb row are always structured index, thumb
        except Exception:
            best_prompt_1 = [0, 0, 0, 0]
        try:
            best_prompt_2 = bboxs[prompt[1]]['boxes'][torch.argmax(bboxs[prompt[1]]['scores'])]
            best_prompt_2 = [float(i)/1000 for i in best_prompt_2]
            print(best_prompt_2)
        except Exception:
            best_prompt_2 = [0, 0, 0, 0]
        
        if 'index' in target_keys:
            return {'index': best_prompt_1, 'thumb': best_prompt_2}
        elif 'hand1' in target_keys:
            return {'hand1': best_prompt_1, 'hand2': best_prompt_2}
        else:
            assert 0 > 1, (print(target_keys, print(row['bboxes']), print(f'If it is a two handed object its gotta be either index-thumb or hand1-hand2')))
    elif len(prompt) == 1:
        try:
            best_prompt_1 = bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])]
        except Exception:
            best_prompt_1 = np.array([0, 0, 0, 0])
        if 'index' in target_keys:
            return {'index': best_prompt_1}
        elif 'handle' in target_keys:
            return {'hand': best_prompt_1}
        else:
            assert 0 > 1, (print(target_keys), print(row['bboxes']), print('If it is a one handed object, then the annotaiton needs to be either hand or index'))
    else:
        assert 0 > 1, (print('target_keys is not an expected length'), print(target_keys), print(len(target_keys)), print(row['bboxes']))


def calculate_iou_results(experiment: VisionExperiment|GeminiExperiment|OWLv2, pred_box_dict: dict, row: dict):
    target_boxes = ast.literal_eval(row['bboxes'])
    target_keys = list(target_boxes.keys())
    target_keys: list[str] = [i.strip() for i in target_keys]
    pred_boxes_reformatted = {}
    if isinstance(experiment, GeminiExperiment):
        pred_boxes_reformatted: dict[str, list] = {key: [pred[1]/1000, pred[0]/1000, pred[3]/1000, pred[2]/1000] for key, pred in pred_box_dict.items()}
    else:
        pred_boxes_reformatted = pred_box_dict
    if len(target_keys) == 2:
        if 'index' in target_keys:
            #in this case, we only need to compare key by key
            index_iou = get_iou(pred_boxes_reformatted['index'], target_boxes['index'])
            thumb_iou = get_iou(pred_boxes_reformatted['thumb'], target_boxes['thumb'])
            return pred_boxes_reformatted, {'index': index_iou, 'thumb': thumb_iou}
        elif 'hand1' in target_keys:
            calc_ious = []
            for pred in pred_boxes_reformatted.values():
                for gt in target_boxes.values():
                    calc_ious.append(get_iou(pred, [val for val in gt.values()]))
                    #gt is a dictioanry structured like this {x1: y1: x2: y2:}
            iou_1_0 = calc_ious[0]
            iou_1_1 = calc_ious[1]
            iou_2_0 = calc_ious[2]
            iou_2_1 = calc_ious[3]
            assignment_scores = [
                (iou_1_0 + iou_2_1, 'pred1_to_hand1'),
                #pred1->gt1, pred2->gt2
                (iou_1_1 + iou_2_0, 'pred1_to_hand2')
                #pred1->gt2, pred2->gt1
            ]
            _, best_assignment = max(assignment_scores)
            print(f'{best_assignment=}')
            pred_boxes_reformatted_2 = {}
            ious = {}
            #we want the correct corresponding keys for our csv
            if best_assignment == 'pred1_to_hand1':
                pred_boxes_reformatted_2['hand1'] = pred_boxes_reformatted['hand1']
                pred_boxes_reformatted_2['hand2'] = pred_boxes_reformatted['hand2']
                ious['hand1'] = iou_1_0
                ious['hand2'] = iou_2_1
                print(f'1: {ious=}')
            else:
                pred_boxes_reformatted_2['hand1'] = pred_boxes_reformatted['hand2']
                pred_boxes_reformatted_2['hand2'] = pred_boxes_reformatted['hand1']
                ious['hand1'] = iou_2_0
                ious['hand2'] = iou_1_1
            print(f'{pred_boxes_reformatted_2=}', f'{ious=}')
            return pred_boxes_reformatted_2, ious
    else:
        if 'index' in target_keys:
            iou = get_iou(pred_boxes_reformatted['index'], [float(val) for val in target_boxes['index'].values()])
            result_iou_dict = {'index': iou}
        elif 'handle' in target_keys:
            iou = get_iou(pred_boxes_reformatted['hand'], [float(val) for val in target_boxes['handle'].values()])
            result_iou_dict = {'hand': iou}
        else:
            assert 0 > 1, (print(target_keys), print(row['bboxes']), print('If it is a one handed object, then the annotaiton needs to be either hand or index'))
        print(f'{pred_boxes_reformatted=}', f'{result_iou_dict=}')
        return pred_boxes_reformatted, result_iou_dict

def get_token_input_output_size(experiment, response):
    if isinstance(experiment, GeminiExperiment):
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
    elif isinstance(experiment, GrokExperiment):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
    elif isinstance(experiment, ClaudeExperiment) or isinstance(experiment, GPTExperiment):
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
    else:
        assert 0 > 1, print(f"Experiment is not one on the expectd types: {GeminiExperiment, VisionExperiment} and is instead {type(experiment)}")
    return input_tokens, output_tokens

def get_iou(bbox_a, bbox_b):
    if len(bbox_a) < 4:
        bbox_a = [0, 0, 0, 0]
    if len(bbox_b) < 4:
        bbox_b = [0, 0, 0 ,0]
    x1_1, y1_1, x2_1, y2_1 = bbox_a
    x1_2, y1_2, x2_2, y2_2 = bbox_b    
    #get intersection coordinates
    x1_1, y1_1, x2_1, y2_1 = float(x1_1), float(y1_1), float(x2_1), float(y2_1)
    x1_2, y1_2, x2_2, y2_2 = float(x1_2), float(y1_2), float(x2_2), float(y2_2)

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    #get intersection area
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection_area = 0
    else:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    #get union area
    union_area = area1 + area2 - intersection_area
    #ge IoU
    if union_area == 0:
        return 0.0
    iou = intersection_area / union_area
    return iou

def get_file_check(file):
    result = re.split(r'_jpg|_JPG|_jpeg|_JPEG|_cleanup|\.rf', file)[0]
    #basically, since we reexported the dataset after changing all of the green stars, all of the .rf extensions got changed, so now, im just checking to see cooresponding paths.
    assert result is not None, print('file check gone wrong')
    return result


def sanitize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '')
    text = text.replace('""', '')
    text = text.replace('""""', '')  
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace(';', '')
    text = re.sub(r'\s+', ' ', text)
    return f"{text.strip()}"

# def fix_none_found(model_list):
#     counter = 0
#     for path in model_list:
#         rows = []
#         with open(path, 'r') as f:
#             reader = csv.DictReader(f, delimiter=';')
#             for row in reader:
#                 if 'none_found' in (list(ast.literal_eval(row['pred_bboxes']).keys())):
#                     keys = [i.strip() for i in list(ast.literal_eval(row['target_bboxes']).keys())]
#                     fixed_pred_bbox = {}
#                     fixed_ious = {}
#                     for i in keys:
#                         fixed_pred_bbox[i] = [0, 0, 0, 0]
#                         fixed_ious[i] = 0.0
#                     new_row = {'img_id': row['img_id'], 'img_path': row['img_path'], 'text_output': row['text_output'], 'pred_bboxes': fixed_pred_bbox, 'target_bboxes': row['target_bboxes'], 'ious': fixed_ious, 'input_tokens': row['input_tokens'], 'output_tokens': row['output_tokens']}
#                 else:
#                     new_row = row
#                 rows.append(new_row)     
#         print(rows)
#         print(len(rows))
#         df = pd.DataFrame(rows)
#         new_path = Path(path).stem
#         print(new_path)
#         new_path += '_test.csv'
#         print(new_path)
#         df.to_csv(f'results/{new_path}', sep=';', encoding='utf-8', index=False)
#         breakpoint()               
#     print(counter)

if __name__ == "__main__":
    model_list = ['results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307.csv', 'results/gemini-2.5-flash-lite-preview-06-17.csv',
                                      'results/gemini-2.5-flash.csv', 'results/gemini-2.0-flash-lite.csv', 'results/gpt-4.1-mini.csv', 'results/gpt-4.1-nano.csv',
                                      'results/grok-2-vision-1212.csv', 'results/o4-mini.csv', 'results/owl_vit.csv', 'results/yolo_uniow.csv', 'results/yolo_world.csv']

    experiment_list = [ClaudeExperiment('claude-3-5-haiku-latest', get_prompt), ClaudeExperiment('claude-3-haiku-20240307', get_prompt),
                       GeminiExperiment('gemini-2.0-flash-lite', get_prompt), GeminiExperiment('gemini-2.5-flash-lite-preview-06-17', get_prompt),
                       GeminiExperiment('gemini-2.5-flash', get_prompt), GPTExperiment('gpt-4.1-mini', get_prompt), GPTExperiment('gpt-4.1-nano', get_prompt),
                       GrokExperiment('grok-2-vision-1212', get_prompt), GPTExperiment('o4-mini', get_prompt), OWLv2()]
    run_grasp_vlm_experiment(experiment_list)
    # fix_none_found(model_list)


    