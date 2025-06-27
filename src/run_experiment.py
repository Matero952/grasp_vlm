from agents.claude_grok import ClaudeExperiment, GrokExperiment
from agents.gemini import GeminiExperiment
from agents.claude_grok import GPTExperiment
from prompt import *
import os
import pandas as pd
import csv
import regex as re
import ast
import time
import numpy as np
from agents.owl import *
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import json
from agents.claude_grok import VisionExperiment
tool_dict_for_yolo_1 = {'drill' : 'drill tool', 'wacker' : 'weed wacker', 'glue' : 'glue gun tool', 'saw' : 'circular saw', 'nail' : 'nail gun', 
    'screwdriver' : 'screwdriver', 'wrench' : 'wrench tool', 'solder' : 'solder iron tool', 'allen' : 'allen wrench tool', 'hammer' : 'hammer'}
tool_regex = r'drill|wacker|glue|saw|nail|screwdriver|wrench|solder|allen|hammer'

def get_prompt_owl(ground_truth_row):
    row = ground_truth_row
    if len(ast.literal_eval(row['bboxes']).keys()) == 2:
        if row['tool'] in ['bowling ball', 'violin bow', 'syringe', 'dart', 'pair of scissors']:
            #index thumb finger tool
            prompt_1 = f"Placement for index finger on the {row['tool']}"
            prompt_2 = f"Where another "
        else:
            #two handed tool
            prompt_1 = f'Where one entire hand can grab the {row['tool']} safely'
            prompt_2 = f'Where another entire hand can grab the {row['tool']} safely'
            match = re.search(r'-?\d+\.?\d*', row['img_path'])
            if not match:
                print(f"No number found.")
                assert 0 > 1
            file_numb = int(match.group(0))
            if file_numb > 10:
                #green star is closer to left hand
                prompt_1 = f"Best placement near the green star for the entire left hand on the {row['tool']}"
                prompt_2 = f"Best placement for the entire right hand on the {row['tool']}"
            elif file_numb < 11:
                prompt_1 = f"Best placement for the entire left hand on the {row['tool']}"
                prompt_2 = f"Best placement near the green star for the entire right hand on the {row['tool']}"
        return (prompt_1, prompt_2)
    else:
        if row['tool'] in ['allen key', 'hammer', 'screwdriver', 'wrench', 'soldering iron'] or 'handle' in row['tool']:
            #handle tools:
            prompt = f"Where an entire hand can grab the {row['tool']} safely"
        else:
            prompt = f"Where the index finger can press on the {row['tool']} safely"
        return prompt
    
def get_test_info_for_prompts(prompt_idx,
                              ground_truth_file='/home/mateo/Github/grasp_vlm/ground_truth_test.csv'):
    experiment = OWLv2()

    os.makedirs('results', exist_ok=True)
    new_df_path = os.path.join('results', f'{prompt_idx}_owl.csv')
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path, sep=';')
    else:
        df = pd.DataFrame(columns=['img_id', 'img_path', 'pred_bboxes', 'target_bboxes', 'ious', 'prompts'])
    config_file = "configs/pretrain/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py"
    checkpoint = 'pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth'
    gt_df = pd.read_csv(ground_truth_file, sep=';', encoding='utf-8')
    gt_df.columns = gt_df.columns.str.replace('"', '', regex=False)
    for index, row in gt_df.iterrows():
        if str(row['img_id']) in df['img_id'].astype(str).values:
            continue
        else:
            prompts = get_prompt(row)
            if isinstance(prompts, tuple):
                prompt_1, prompt_2 = prompts
                prompt = [prompt_1, prompt_2]
                #index,thumb or left,right
            else:
                prompt = [prompts]
            assert all(isinstance(i, str) for i in prompt), print(repr(prompt))
            img = str(os.path.join('/home/mateo/Github/grasp_vlm', row['img_path']))
            img = Image.open(img)
            img = T.ToTensor()(img)
            print(prompt)
            # breakpoint()
            bboxs = experiment.predict(img, prompt)
            print(torch.argmax(bboxs[prompt[0]]['scores']))
            print(bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])])
            # breakpoint()
            if len(prompt) > 1:
                try:
                    best_prompt_1 = bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])]
                except Exception as e:
                    print(e)
                    best_prompt_1 = np.array([0, 0, 0, 0])
                try:
                    best_prompt_2 = bboxs[prompt[1]]['boxes'][torch.argmax(bboxs[prompt[1]]['scores'])]
                except Exception as e:
                    print(e)
                    best_prompt_2 = np.array([0, 0, 0, 0])
                if 'left' in prompt_1:
                    best_pred = {'left': [i/1000 for i in best_prompt_1.tolist()], 'right': [i/1000 for i in best_prompt_2.tolist()]}
                elif 'index' in prompt_1:
                    best_pred = {'index': [i/1000 for i in best_prompt_1.tolist()], 'thumb': [i/1000 for i in best_prompt_2.tolist()]}
                else:
                    assert 0 > 1
            else:
                try:
                    best_prompt= bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])]
                except Exception as e:
                    print(e)
                    best_prompt = np.array([0, 0, 0, 0])
                if 'hand' in prompt[0]:
                    best_pred = {'hand': [i/1000 for i in best_prompt.tolist()]}
                elif 'index' in prompt[0]:
                    best_pred = {'index': [i/1000 for i in best_prompt.tolist()]}
                else:
                    assert 0 > 1
            print(f'{best_pred=}')
            ious = {}
            pred_bboxes_reformat = {}
            bboxes = ast.literal_eval(row['bboxes'])
            for key, pred_bbox in best_pred.items():
                #vlms supposed to output in key value pairs, so im just checking the values of cooresponding keys
                assert isinstance(key, str)
                switch = False
                for gt_key in bboxes.keys():
                    if key.lower().strip() in gt_key:
                        cooresponding_gt_bbox = bboxes[gt_key]
                        assert isinstance(cooresponding_gt_bbox, dict)
                        cooresponding_gt_bbox = list(cooresponding_gt_bbox.values())
                        switch = True
                if not switch:
                    print(f'Model produced a bad key')
                    print(key)
                    cooresponding_gt_bbox = [0, 0, 0 ,0]
                print(pred_bbox)
                print(cooresponding_gt_bbox)
                pred_bboxes_reformat[key] = pred_bbox
                iou = get_iou(pred_bbox, cooresponding_gt_bbox)
                print(iou)
                print(f'{pred_bbox=}')
                print(f'{cooresponding_gt_bbox=}')
                # breakpoint()
                #iou assumes consistent format
                ious[key.lower().strip()] = iou
                print(iou)
                # breakpoint()
            df.loc[len(df)] = [row['img_id'], row['img_path'], str(pred_bboxes_reformat), row['bboxes'], ious, sanitize_text(str(prompt))]
            df.to_csv(new_df_path, sep=';', encoding='utf-8', index=False)

            # breakpoint()


def rerun_experiment(experiment: GeminiExperiment|VisionExperiment|None, ground_truth_csv_path='ground_truth_test.csv', to_change:str = None):
    if experiment is None:
        owl_experiment = True
    else:
        owl_experiment = False
    os.makedirs('results', exist_ok=True)
    file_stem = f'{experiment.model}' 
    file_stem += f'_change_{to_change}' if to_change is not None else f''
    file_stem += '.csv'
    new_df_path = os.path.join('results', file_stem)



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
    #we dont need to keep on trying to get a response out of owl if it doesnt work because it is deterministic, if that img and prompt do not work, then they will never work.
    target_keys = list(ast.literal_eval(row['bboxes']).keys())
    target_keys: list[str] = [i.strip() for i in target_keys]
    if len(prompt) == 2:
        try:
            best_prompt_1  = bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])]
            #the prompts if its an index-thumb row are always structured index, thumb
        except Exception:
            best_prompt_1 = np.array([0, 0, 0, 0])
        try:
            best_prompt_2 = bboxs[prompt[1]]['boxes'][torch.argmax(bboxs[prompt[1]]['scores'])]
        except Exception:
            best_prompt_2 = np.array([0, 0, 0, 0])
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
                    calc_ious.append(pred, [val for val in gt.values()])
                    #gt is a dictioanry structured like this {x1: y1: x2: y2:}
            iou_1_0 = calc_ious[0]
            iou_1_1 = calc_ious[1]
            iou_2_0 = calc_ious[2]
            iou_2_1 = calc_ious[3]
            assignment_scores = [
                (iou_1_0 + iou_2_1, 'pred1_to_hand1'),  # pred1->gt1, pred2->gt2
                (iou_1_1 + iou_2_0, 'pred1_to_hand2')   # pred1->gt2, pred2->gt1
            ]
            best_score, best_assignment = max(assignment_scores)
            print(f'{best_assignment=}')
            pred_boxes_reformatted_2 = {}
            ious = {}
            #reinitialize it so that we can now get the correct, cooresponding keys
            if best_assignment == 'pred1_to_hand1':
                pred_boxes_reformatted_2['hand1'] = pred_boxes_reformatted['hand1']
                pred_boxes_reformatted_2['hand2'] = pred_boxes_reformatted['hand2']
                ious['hand1'] = iou_1_0
                ious['hand2'] = iou_2_1
                print(f'1: {ious=}')  # Fixed syntax
            else:
                pred_boxes_reformatted_2['hand1'] = pred_boxes_reformatted['hand2']
                pred_boxes_reformatted_2['hand2'] = pred_boxes_reformatted['hand1']
                ious['hand1'] = iou_2_0
                ious['hand2'] = iou_1_1
            return pred_boxes_reformatted_2, ious
        else:
            if 'index' in target_keys:
                iou = get_iou(pred_boxes_reformatted['index'], target_boxes['index'])
                result_iou_dict = {'index': iou}
            elif 'handle' in target_keys:
                iou = get_iou(pred_boxes_reformatted['hand'], target_boxes['handle'])
                result_iou_dict = {'hand': iou}
            else:
                assert 0 > 1, (print(target_keys), print(row['bboxes']), print('If it is a one handed object, then the annotaiton needs to be either hand or index'))
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
        assert 0 > 1, print(f"Experiment is not one on the expectd types: {GeminiExperiment, GrokExperiment, ClaudeExperiment, GPTExperiment} and is instead {type(experiment)}")
    return input_tokens, output_tokens

def get_iou(bbox_a, bbox_b):
    if len(bbox_a) < 4:
        bbox_a = [0, 0, 0, 0]
    if len(bbox_b) < 4:
        bbox_b = [0, 0, 0 ,0]
    x1_1, y1_1, x2_1, y2_1 = bbox_a
    x1_2, y1_2, x2_2, y2_2 = bbox_b    
    #get intersection coordinates
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

def sanitize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '')
    text = text.replace('""', '')
    text = text.replace('""""', '')  
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace(';', '')
    text = re.sub(r'\s+', ' ', text)
    return f"{text.strip()}"

def reformat_owl(box, from_size_tensor, owl_tensor):
    #converts (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    #to (bot_x1, bot_y1, top_x2, top_y2)'
    assert isinstance(from_size_tensor, torch.Tensor)
    assert isinstance(owl_tensor, torch.Size)
    x1, y1, x2, y2 = box
    _, og_height, og_width = from_size_tensor.shape
    print(owl_tensor)
    print(f"og height: {og_height}, og_width: {og_width}")
    breakpoint()
    _, _, owl_height, owl_width = owl_tensor
    print(f"owl height: {owl_height}, owl_width: {owl_width}")

    fx = owl_width / og_width
    print(fx)
    fy = owl_height / og_height
    print(fy)
    return [x1 * fx, y1 * fy, x2 * fx, y2 * fy]

if __name__ == "__main__":
    # print(get_iou([0.11, 0.34, 0.47, 0], [0.372, 0.137, 0.49, 0.4225]))
    # print(get_iou([0.32, 0.14, 0.47, 0.6], [0.372, 0.137, 0.49, 0.4225]))
    # print(get_iou([0.1, 0.37, 0.52, 0.3], [0.119, 0.35, 0.2605, 0.406]))
    # print(get_iou([0.32, 0.14, 0.47, 0.6], [0.119, 0.35, 0.2605, 0.406]))
    # breakpoint()
    
    # model_list = ['results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307.csv', 'results/gemini-2.5-flash-lite-preview-06-17.csv',
    #                                   'results/gemini-2.5-flash.csv', 'results/gemini-2.0-flash-lite.csv', 'results/gpt-4.1-mini.csv', 'results/gpt-4.1-nano.csv',
    #                                   'results/grok-2-vision-1212.csv', 'results/o4-mini.csv', 'results/owl_vit_prompt_1.csv', 'results/yolo_uniow_prompt_1.csv', 'results/yolo_world_prompt_1.csv']
    # model_list = [Path(i).stem for i in model_list]
    with open('ground_truth_test.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            print(process_vlm_output(GeminiExperiment('gemini-2.5-flash-lite-preview-06-17', get_prompt), row))
            breakpoint()
    # rerun_experiment(GeminiExperiment('gemini-2.5-flash-lite-preview-06-17', get_prompt), to_change='two_handed')

    # rereun_experiment(GeminiExperiment('gemini-2.5-flash-lite-preview-06-17', get_prompt))
# def rereun_experiment(experiment, ground_truth_csv_path='ground_truth_test.csv'):
#     os.makedirs('results', exist_ok=True)
#     new_df_path = os.path.join('results', f'{experiment.model}_2.csv')
#     to_replace = ['chainsaw', 'bolt_cutters', 'shovel', 'multimeter', 'can_opener']
#     if os.path.exists(f'results/{experiment.model}.csv'):
#         print(True)
#         # breakpoint()
#         read_df = pd.read_csv(f'results/{experiment.model}.csv', sep=';')
#         print(read_df)
#         new_df = pd.DataFrame(columns=['img_id', 'img_path', 'text_output', 'pred_bboxes', 'target_bboxes', 'ious', 'input_tokens', 'output_tokens'])
#         if isinstance(experiment, GeminiExperiment):
#             delay = 5.2
#         else:
#             delay = 0.0
#         counter = 0
#         with open(ground_truth_csv_path, 'r') as f:
#             reader = csv.DictReader(f, delimiter=';')
#             for row in reader:
#                 print(str(row['img_id']).strip())
#                 print(read_df['img_id'].astype(str).values)
#                 # breakpoint()
#                 if str(row['img_id']).strip() in read_df['img_id'].astype(str).values:
#                     print(f'True 2')
#                     # breakpoint()
#                     if any(i in str(row['img_path']).strip() for i in to_replace):
#                         counter += 1
#                         response, new_text  = experiment.process_sample(row['img_path'], row['tool'], row['vlm_role'], 
#                                                                     experiment.model, row['task'], ast.literal_eval(row['bboxes']))
#                         new_text = new_text.replace('\n', '')
#                         new_text = re.sub(r'```json\s*', '```json', new_text)
#                         new_text = re.sub(r'\s*```$', '```', new_text)
#                         max_attempts = 3
#                         for attempt in range(max_attempts):
#                             pred_bboxes = re.search(r'\{(?:[^{}]|(?R))*\}', new_text)
#                             if pred_bboxes is not None:
#                                 pred_bboxes = pred_bboxes.group(0)
#                                 try:
#                                     clean_str = str(pred_bboxes).replace("{{", "[{").replace("}}", "}]")
#                                     pred_bboxes = json.loads(clean_str)
#                                     break
#                                 # pred_bboxes = ast.literal_eval(str(pred_bboxes))
#                                 except ValueError:
#                                     print(f"ValueError malformed node or string")
#                                     print(f'{pred_bboxes=}')
#                                     print(f'{type(pred_bboxes)=}')
#                                     print(new_text)
#                         else:
#                             if len(list(ast.literal_eval(row['bboxes']).keys())) > 1:
#                                 if 'hand1' in list(ast.literal_eval(row['bboxes']).keys()):
#                                     pred_bboxes = {'hand1': [0.0, 0.0, 0.0, 0.0], 'hand2': [0.0, 0.0, 0.0, 0.0]}
#                                 elif 'index' in list(ast.literal_eval(row['bboxes']).keys()):
#                                     pred_bboxes = {'index': [0.0, 0.0, 0.0, 0.0], 'thumb': [0.0, 0.0, 0.0, 0.0]}
#                                 else:
#                                     print('wtf')
#                                     breakpoint()
#                             else:
#                                 if 'handle' in list(ast.literal_eval(row['bboxes']).keys()):
#                                     pred_bboxes = {'hand': [0.0, 0.0, 0.0, 0.0]}
#                                 elif 'index' in list(ast.literal_eval(row['bboxes']).keys()):
#                                     pred_bboxes = {'index': [0.0, 0.0, 0.0, 0.0]}
#                                 else:
#                                     print('wtf2')
#                                     breakpoint()
#                         assert isinstance(pred_bboxes, dict), print(repr(pred_bboxes), type(pred_bboxes), repr(new_text))
#                 # print(text)
#                         ious = {}
#                         pred_bboxes_reformat = {}
#                         bboxes = ast.literal_eval(row['bboxes'])
#                         print(pred_bboxes)
#                         calc_ious = []
#                         for pred in pred_bboxes.values():
#                             for gt in bboxes.values():
#                                 if isinstance(experiment, GeminiExperiment):
#                                     pred = [pred[1]/1000, pred[0]/1000, pred[3]/1000, pred[2]/1000]
#                                 print(gt)
#                                 calc_ious.append(get_iou(pred, [i for i in gt.values()]))
#                                 print(f'{get_iou(pred, [i for i in gt.values()])=}')
#                         if sum(calc_ious) == 0:
#                             pred_bboxes_reformat['hand1'] = list(pred_bboxes.values())[0]
#                             pred_bboxes_reformat['hand2'] = list(pred_bboxes.values())[1]
#                             iou_1_0 = calc_ious[0]
#                             iou_1_1 = calc_ious[1]
#                             iou_2_0 = calc_ious[2]
#                             iou_2_1 = calc_ious[3]
#                             assignment_scores = [
#                                 (iou_1_0 + iou_2_1, 'pred1_to_hand1'),  # pred1->gt1, pred2->gt2
#                                 (iou_1_1 + iou_2_0, 'pred1_to_hand2')   # pred1->gt2, pred2->gt1
#                             ]
#                         else:
#                             pred1 = list(pred_bboxes.values())[0]
#                             pred2 = list(pred_bboxes.values())[1]
#                             iou_1_0 = calc_ious[0]  # pred1 vs gt1
#                             iou_1_1 = calc_ious[1]  # pred1 vs gt2
#                             iou_2_0 = calc_ious[2]  # pred2 vs gt1
#                             iou_2_1 = calc_ious[3]  # pred2 vs gt2
#                             assignment_scores = [
#                                 (iou_1_0 + iou_2_1, 'pred1_to_hand1'),  # pred1->gt1, pred2->gt2
#                                 (iou_1_1 + iou_2_0, 'pred1_to_hand2')   # pred1->gt2, pred2->gt1
#                             ]

#                             # Choose the assignment with higher total IoU
#                         best_score, best_assignment = max(assignment_scores)
#                         print(f'{best_assignment=}')

#                         if best_assignment == 'pred1_to_hand1':
#                             if isinstance(experiment, GeminiExperiment):
#                                 pred_bboxes_reformat['hand1'] = [pred1[1]/1000, pred1[0]/1000, pred1[3]/1000, pred1[2]/1000]
#                                 pred_bboxes_reformat['hand2'] = [pred2[1]/1000, pred2[0]/1000, pred2[3]/1000, pred2[2]/1000]
#                             else:
#                                 pred_bboxes_reformat['hand1'] = pred1
#                                 pred_bboxes_reformat['hand2'] = pred2
#                             ious['hand1'] = iou_1_0
#                             ious['hand2'] = iou_2_1
#                             print(f'1: {ious=}')  # Fixed syntax
#                         else:
#                             if isinstance(experiment, GeminiExperiment):
#                                 pred_bboxes_reformat['hand1'] = [pred2[1]/1000, pred2[0]/1000, pred2[3]/1000, pred2[2]/1000]
#                                 pred_bboxes_reformat['hand2'] = [pred1[1]/1000, pred1[0]/1000, pred1[3]/1000, pred1[2]/1000]
#                             else:
#                                 pred_bboxes_reformat['hand1'] = pred2
#                                 pred_bboxes_reformat['hand2'] = pred1
#                             ious['hand1'] = iou_2_0
#                             ious['hand2'] = iou_1_1
#                             print(f'2: {ious=}')  # Fixed syntax
#                         print(ious)
#                         if not ious:
#                             assert 0 > 1


#                         pred_bboxes_reformat = dict(sorted(pred_bboxes_reformat.items()))
#                         ious = dict(sorted(ious.items()))
#                         print(pred_bboxes_reformat)
#                         print(ious)
#                         # breakpoint()
#                         if isinstance(experiment, GeminiExperiment):
#                             input_tokens = response.usage_metadata.prompt_token_count
#                             output_tokens = response.usage_metadata.candidates_token_count
#                         elif isinstance(experiment, GPTExperiment):
#                             input_tokens = response.usage.input_tokens
#                             output_tokens = response.usage.output_tokens
#                         elif isinstance(experiment, GrokExperiment):
#                             input_tokens = response.usage.prompt_tokens
#                             output_tokens = response.usage.completion_tokens
#                         elif isinstance(experiment, ClaudeExperiment):
#                             input_tokens = response.usage.input_tokens
#                             output_tokens = response.usage.output_tokens
#                         new_df.loc[len(new_df)] = [row['img_id'], row['img_path'], sanitize_text(new_text), pred_bboxes_reformat, row['bboxes'], ious, input_tokens, output_tokens]
#                         new_df.to_csv(new_df_path, sep=';', encoding='utf-8', index=False)
#                         time.sleep(delay)
#                 else:
#                     to_check_row = read_df[read_df['img_id'].astype(str).str.strip() == row['img_id'].strip()].iloc[0]
#                     new_df.loc[len(new_df)] = [row['img_id'], row['img_path'], to_check_row['text_output'], to_check_row['pred_bboxes'], row['bboxes'], to_check_row['ious'], to_check_row['input_tokens'], to_check_row['output_tokens']]
#             else:
#                 print('wtf3')
#                 breakpoint()

# def run_experiment(experiment, ground_truth_csv_path):
#     os.makedirs('results', exist_ok=True)
#     new_df_path = os.path.join('results', f'{experiment.model}.csv')
#     if os.path.exists(new_df_path):
#         df = pd.read_csv(new_df_path, sep=';', encoding='utf-8')
#         df.columns = df.columns.str.replace('"', '', regex=False)
#     else:
#         df = pd.DataFrame(columns=['img_id', 'img_path', 'text_output', 'pred_bboxes', 'target_bboxes', 'ious', 'input_tokens', 'output_tokens'])
#     if isinstance(experiment, GeminiExperiment):
#         delay = 5.2
#     else:
#         delay = 0
#     with open(ground_truth_csv_path) as f:
#         reader = csv.DictReader(f, delimiter=';')
#         for row in reader:
#             if str(row['img_id']) in df['img_id'].astype(str).values:
#                 print(f"Skipping: {row['img_id']}")
#                 continue
#             else:
#                 file_path = row['img_path']
#                 tool = row['tool']
#                 vlm_role = row['vlm_role']
#                 model = experiment.model
#                 task = row['task']
#                 bboxes = ast.literal_eval(row['bboxes'])
#                 response, text  = experiment.process_sample(file_path, tool, vlm_role, model, task, bboxes)
#                 text = text.replace('\n', '')
#                 text = re.sub(r'```json\s*', '```json', text)
#                 text = re.sub(r'\s*```$', '```', text)
#                 max_attempts = 3
#                 for attempt in range(max_attempts):
#                     pred_bboxes = re.search(r'\{(?:[^{}]|(?R))*\}', text)
#                     if pred_bboxes is not None:
#                         pred_bboxes = pred_bboxes.group(0)
#                         try:
#                             clean_str = str(pred_bboxes).replace("{{", "[{").replace("}}", "}]")
#                             pred_bboxes = json.loads(clean_str)
#                             break
#                         # pred_bboxes = ast.literal_eval(str(pred_bboxes))
#                         except ValueError:
#                             print(f"ValueError malformed node or string")
#                             print(f'{pred_bboxes=}')
#                             print(f'{type(pred_bboxes)=}')
#                             print(text)
#                 else:
#                     pred_bboxes = {'none_found': [0.0, 0.0, 0.0, 0.0], 'none_found': [0.0, 0.0, 0.0, 0.0]}
#                     print(f"MAX ATTEMPT LIMIT: {max_attempts}, REACHED LOL")
#                 # pred_bboxes = ast.literal_eval(pred_bboxes)

#                 print(f'{pred_bboxes}=')
#                 # print(f'{list(pred_bboxes.keys())=}')
#                 # breakpoint()
#                 assert isinstance(pred_bboxes, dict), print(repr(pred_bboxes), type(pred_bboxes), repr(text))
#                 # print(text)
#                 ious = {}
#                 pred_bboxes_reformat = {}
#                 for key, pred_bbox in pred_bboxes.items():
#                     #vlms supposed to output in key value pairs, so im just checking the values of cooresponding keys
#                     assert isinstance(key, str)
#                     switch = False
#                     for gt_key in bboxes.keys():
#                         if key.lower().strip() in gt_key:
#                             cooresponding_gt_bbox = bboxes[gt_key]
#                             assert isinstance(cooresponding_gt_bbox, dict)
#                             cooresponding_gt_bbox = list(cooresponding_gt_bbox.values())
#                             switch = True
#                     if not switch:
#                         print(f'Model produced a bad key')
#                         print(key)
#                         cooresponding_gt_bbox = [0, 0, 0 ,0]
#                     print(f'{cooresponding_gt_bbox=}')
#                     # breakpoint()

#                     if isinstance(experiment, GeminiExperiment):
#                         pred_bbox = [pred_bbox[1]/1000, pred_bbox[0]/1000, pred_bbox[3]/1000, pred_bbox[2]/1000]
#                         #switch from gemini format of yxyz to xyxy and renormalize from 0 - 1000 to 0 - 1
#                         # cooresponding_gt_bbox = [cooresponding_gt_bbox[1], cooresponding_gt_bbox[0], 
#                         #                        cooresponding_gt_bbox[3], cooresponding_gt_bbox[2]]
#                         #switch to gemini format from yx yx to xy xy
#                     else:
#                         pass
#                     pred_bboxes_reformat[key] = pred_bbox
#                     iou = get_iou(pred_bbox, cooresponding_gt_bbox)
#                     print(iou)
#                     print(f'{pred_bbox=}')
#                     print(f'{cooresponding_gt_bbox=}')
#                     # breakpoint()
#                     #iou assumes consistent format
#                     ious[key.lower().strip()] = iou
#                 if isinstance(experiment, GeminiExperiment):
#                     input_tokens = response.usage_metadata.prompt_token_count
#                     output_tokens = response.usage_metadata.candidates_token_count
#                 elif isinstance(experiment, GPTExperiment):
#                     input_tokens = response.usage.input_tokens
#                     output_tokens = response.usage.output_tokens
#                 elif isinstance(experiment, GrokExperiment):
#                     input_tokens = response.usage.prompt_tokens
#                     output_tokens = response.usage.completion_tokens
#                 elif isinstance(experiment, ClaudeExperiment):
#                     input_tokens = response.usage.input_tokens
#                     output_tokens = response.usage.output_tokens
#                 df.loc[len(df)] = [row['img_id'], row['img_path'], sanitize_text(text), pred_bboxes_reformat, row['bboxes'], ious, input_tokens, output_tokens]
#                 df.to_csv(new_df_path, sep=';', encoding='utf-8', index=False)
#                 time.sleep(delay)
#             breakpoint()



    