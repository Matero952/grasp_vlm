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
tool_dict_for_yolo_1 = {'drill' : 'drill tool', 'wacker' : 'weed wacker', 'glue' : 'glue gun tool', 'saw' : 'circular saw', 'nail' : 'nail gun', 
    'screwdriver' : 'screwdriver', 'wrench' : 'wrench tool', 'solder' : 'solder iron tool', 'allen' : 'allen wrench tool', 'hammer' : 'hammer'}
tool_regex = r'drill|wacker|glue|saw|nail|screwdriver|wrench|solder|allen|hammer'
def generate_owl_prompts(base_directory):
    #generates owl prompts from old results
    owl_paths = []
    prompts = []
    path = Path(base_directory)
    for i in path.rglob('*.csv'):
        print(str(i))
        if 'owl' in str(i):
            print(str(i))
            owl_paths.append(str(i))
        else:
            continue
    assert len(owl_paths) == 13
    for i in owl_paths:
        assert isinstance(i, str)
        df = pd.read_csv(i, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False)
        to_check_row_hand = df.iloc[0]
        to_check_row_index = df.iloc[1]
        hand_prompt = to_check_row_hand['noun']
        index_prompt = to_check_row_index['noun']
        prompts.append((index_prompt, hand_prompt))
    print(prompts)
    return prompts

def run_owl_experiments(experiment, ground_truth_csv, prompt_base_dir = 'results'):
    prompts = generate_owl_prompts(prompt_base_dir)
    print(f"Starting OWL experiments for {prompts=}")
    counter = 0
    completed_experiment_list = []
    for i in prompts:
        os.makedirs('results', exist_ok=True)
        save_dir = os.path.join("results", 'owl')
        os.makedirs(save_dir, exist_ok=True)
        new_df_path = os.path.join(save_dir, f"owlvit-base-patch32_{counter}.csv")
        df = pd.DataFrame(columns=['img_id', 'img_path', 'pred_bbox', 'noun', 'target_bbox', 'iou'])
        df.columns = df.columns.str.replace('"', '', regex=False)
        with open(ground_truth_csv) as f:
            reader = csv.DictReader(f, delimiter=';')
            # reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                img = Image.open(str(row['img_path']))
                img = T.ToTensor()(img)
                prompt_index, prompt_hand = i
                tool_match = re.search(tool_regex, row['img_path'])
                if tool_match:
                    tool = tool_match.group(0)
                else:
                    raise ValueError('No tool found in the image path.')
                if 'index' in str(row['annotation_type']):
                    prompt = f"{prompt_index} on the {tool_dict_for_yolo_1[tool]}"
                elif 'four' in str(row['annotation_type']):
                    prompt = f"{prompt_hand} on the {tool_dict_for_yolo_1[tool]}"
                else:
                    raise ValueError('No recognized annotation type')
                print(f'{prompt=}')
                prompt = [f'{prompt}']
                bboxs = experiment.predict(img, prompt)
                print(f'{bboxs=}')
                best_box = bboxs[prompt[0]]['boxes'][torch.argmax(bboxs[prompt[0]]['scores'])]
                best_box = best_box.tolist()
                iou = get_iou(best_box, ast.literal_eval(row['bbox']))
                df.loc[int(row['img_id'])] = [row['img_id'], row['img_path'], str(best_box).strip(), str(prompt), row['bbox'], iou]
                df.to_csv(new_df_path, sep=';', encoding='utf-8', index=False)
        print(f"Finished OWL experiment for {i=}")
        print(f"Starting OWL experiment for {i=}")
        counter += 1
        completed_experiment_list.append(new_df_path)
    # plot_box_and_whiskers(get_owl_single(completed_experiment_list, 'src/ground_truth_owl.csv'))


def run_experiment(experiment, ground_truth_csv_path):
    # rows = []
    #{img_id: , img_path: , text_output: , pred_bboxes: , target_bboxes: , ious: , input_tokens: , output_tokens: }
    os.makedirs('results', exist_ok=True)
    new_df_path = os.path.join('results', f'{experiment.model}.csv')
    if os.path.exists(new_df_path):
        df = pd.read_csv(new_df_path, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False)
    else:
        df = pd.DataFrame(columns=['img_id', 'img_path', 'text_output', 'pred_bboxes', 'target_bboxes', 'ious', 'input_tokens', 'output_tokens'])
    if isinstance(experiment, GeminiExperiment):
        delay = 5.2
    else:
        delay = 0
    with open(ground_truth_csv_path) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if str(row['img_id']) in df['img_id'].astype(str).values:
                print(f"Skipping: {row['img_id']}")
                continue
            else:
                file_path = row['img_path']
                tool = row['tool']
                vlm_role = row['vlm_role']
                model = experiment.model
                task = row['task']
                bboxes = ast.literal_eval(row['bboxes'])
                response, text  = experiment.process_sample(file_path, tool, vlm_role, model, task, bboxes)
                text = text.replace('\n', '')
                text = re.sub(r'```json\s*', '```json', text)
                text = re.sub(r'\s*```$', '```', text)
                max_attempts = 3
                for attempt in range(max_attempts):
                    pred_bboxes = re.search(r'\{(?:[^{}]|(?R))*\}', text)
                    if pred_bboxes is not None:
                        pred_bboxes = pred_bboxes.group(0)
                        try:
                            clean_str = str(pred_bboxes).replace("{{", "[{").replace("}}", "}]")
                            pred_bboxes = json.loads(clean_str)
                            break
                        # pred_bboxes = ast.literal_eval(str(pred_bboxes))
                        except ValueError:
                            print(f"ValueError malformed node or string")
                            print(f'{pred_bboxes=}')
                            print(f'{type(pred_bboxes)=}')
                            print(text)
                else:
                    pred_bboxes = {'none_found': [0.0, 0.0, 0.0, 0.0], 'none_found': [0.0, 0.0, 0.0, 0.0]}
                    print(f"MAX ATTEMPT LIMIT: {max_attempts}, REACHED LOL")
                # pred_bboxes = ast.literal_eval(pred_bboxes)

                print(f'{pred_bboxes}=')
                # print(f'{list(pred_bboxes.keys())=}')
                # breakpoint()
                assert isinstance(pred_bboxes, dict), print(repr(pred_bboxes), type(pred_bboxes), repr(text))
                # print(text)
                ious = {}
                pred_bboxes_reformat = {}
                for key, pred_bbox in pred_bboxes.items():
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
                    print(f'{cooresponding_gt_bbox=}')
                    # breakpoint()

                    if isinstance(experiment, GeminiExperiment):
                        pred_bbox = [pred_bbox[1]/1000, pred_bbox[0]/1000, pred_bbox[3]/1000, pred_bbox[2]/1000]
                        #switch from gemini format of yxyz to xyxy and renormalize from 0 - 1000 to 0 - 1
                        # cooresponding_gt_bbox = [cooresponding_gt_bbox[1], cooresponding_gt_bbox[0], 
                        #                        cooresponding_gt_bbox[3], cooresponding_gt_bbox[2]]
                        #switch to gemini format from yx yx to xy xy
                    else:
                        pass
                    pred_bboxes_reformat[key] = pred_bbox
                    iou = get_iou(pred_bbox, cooresponding_gt_bbox)
                    print(iou)
                    print(f'{pred_bbox=}')
                    print(f'{cooresponding_gt_bbox=}')
                    # breakpoint()
                    #iou assumes consistent format
                    ious[key.lower().strip()] = iou
                if isinstance(experiment, GeminiExperiment):
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                elif isinstance(experiment, GPTExperiment):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                elif isinstance(experiment, GrokExperiment):
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                elif isinstance(experiment, ClaudeExperiment):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                df.loc[len(df)] = [row['img_id'], row['img_path'], sanitize_text(text), pred_bboxes_reformat, row['bboxes'], ious, input_tokens, output_tokens]
                df.to_csv(new_df_path, sep=';', encoding='utf-8', index=False)
                time.sleep(delay)
                
                
                
                
# def run_experiment(experiment, ground_truth_csv, delay, iou_tolerance = None):
#     os.makedirs("results", exist_ok=True)
#     # save_dir = os.path.join("results", experiment.model)
#     # os.makedirs(save_dir, exist_ok=True)
#     new_df_path = os.path.join('results', f'{experiment.model}.csv' )

#     if iou_tolerance is not None:
#         correct = 0
#         seen = 0
#     if os.path.exists(new_df_path):
#         df = pd.read_csv(new_df_path, sep=';', encoding='utf-8')
#         df.columns = df.columns.str.replace('"', '', regex=False)
#     else:
#         df = pd.DataFrame(columns=["img_id", "img_path", "text_output", "pred_bboxes", "target_bboxes", "iou", "token_cost"])
#     with open(ground_truth_csv) as f:
#         reader = csv.DictReader(f, delimiter=';')
#         for row in reader:
#             if str(row['img_id']) in df['img_id'].astype(str).values:
#                 if iou_tolerance is not None:
#                     to_check_row = df[df['img_id'] == row['img_id']]
#                     correct += 1 if to_check_row['iou'].iloc[0] > iou_tolerance else 0
#                     seen += 1
#                 print(f"Skipping: {row['img_id']}")
#                 continue
#             else:
#                 text = experiment.process_sample(row['img_path'], row['tool'], row['annotation_type'])
#                 text = sanitize_text(clean_text(text))
#                 pred_bbox = get_pred_bbox(text, int(row['img_id']))
#                 target_bbox = ast.literal_eval(row['bbox'])
#                 if pred_bbox is None:
#                     iou = "No predicted bbox found."
#                 else:
#                     iou = get_iou(pred_bbox, target_bbox)
#                 df.loc[len(df)] = [row['img_id'], row['img_path'], sanitize_text(clean_text(text)), sanitize_text(clean_text(pred_bbox)), sanitize_text(clean_text(target_bbox)), sanitize_text(clean_text(str(iou)))]
#                 if iou_tolerance is not None and pred_bbox is not None:
#                     correct += 1 if iou > iou_tolerance else 0
#                     seen += 1
#                 df.to_csv(new_df_path, index=False, sep=';')
#                 time.sleep(delay)
#             if iou_tolerance is not None:
#                 print(f"img_id: {row['img_id']}, accuracy: {correct / seen}")
#             else:
#                 print(f"img_id: {row['img_id']}")


# #helper functions for extracting key information from responses
# def get_pred_bbox(indv_response, img_id):
#     indv_response = indv_response.replace('""', '"')
#     numbers_match = re.findall(r'\b\d+\.\d+|\b\d+|\B\.\d+', indv_response)
#     if numbers_match:
#         if needs_denormalize(numbers_match[-4:]):
#             #runs check for denormalization because 
#             #some bnd boxes are normalized and some are not
#             bbox = denormalize(img_id, numbers_match[-4:], 'src/ground_truth.csv')
#             #denormalizing if the vlm outputted number normalized 0 - 1
#         else:
#             #otherwise we just take the vlm-outputted bnd box
#             bbox = [float(i) for i in numbers_match[-4:]]
#     else:
#         bbox = [0, 0, 0, 0]
#     return bbox
def get_average_vertex_difference(bbox_a, bbox_b):
    assert len(bbox_a) == 4
    assert len(bbox_b) == 4
    x1_1, y1_1, x2_1, y2_1 = bbox_a
    x1_2, y1_2, x2_2, y2_2 = bbox_b
    #redefine bboxes as 4 points
    bbox_a = [[x1_1, y1_1], [x1_1, y2_1], [x2_1, y2_1], [x2_1, y1_1]]
    #top left, bottom left, bottom right, top right
    bbox_b = [[x1_2, y1_2], [x1_2, y2_2], [x2_2, y2_2], [x2_2, y1_2]]
    #top left, bottom left, bottom right, totp right
    mag_vec_top_left = np.linalg.norm(np.array(bbox_a[0]) - np.array(bbox_b[0]))
    mag_vec_bot_left = np.linalg.norm(np.array(bbox_a[1]) - np.array(bbox_b[1]))
    mag_vec_bot_right = np.linalg.norm(np.array(bbox_a[2]) - np.array(bbox_b[2]))
    mag_vec_top_right = np.linalg.norm(np.array(bbox_a[3]) - np.array(bbox_b[3]))
    #get magnitude of all of the vectors drawn from one point of box a to the coorepsonding pont in box b
    return (mag_vec_top_left + mag_vec_bot_left + mag_vec_bot_right + mag_vec_top_right) / 4
    







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

# def clean_text(text):
#     if not isinstance(text, str):
#         return text
#     # Replace all runs of whitespace (spaces, newlines, tabs) with a single space
#     cleaned = re.sub(r'\s+', ' ', text)
#     return cleaned.strip()

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
    # claude_models = ['claude-3-5-haiku-latest', 'claude-3-haiku-20240307']
    # grok_models = ['grok-2-vision-1212']
    gpt_models = ['gpt-4.1-mini', 'gpt-4.1-nano', 'o4-mini']
    gem_models = ['gemini-2.0-flash-lite']
    # for i in claude_models:
    #     run_experiment(ClaudeExperiment(i, get_prompt), "ground_truth_test.csv")
    
    # # gem_models = ['gemini-2.5-flash-lite-preview-06-17']
    # # gem_models = [('gemini-2.0-flash-lite', 2.5), ('gemini-2.0-flash', 4.5), ('gemini-2.0-flash-lite', 2.5), ('gemini-2.5-flash-preview-05-20', 5.5), ('gemini-1.5-flash', 4.3)]
    # for i in grok_models:
        # run_experiment(GrokExperiment(i, get_prompt), "ground_truth_test.csv")

    # max_attempts = 100
    attempts = 0
    while True:
        try:
            for i in gem_models:
                run_experiment(GeminiExperiment(i, get_prompt), 'ground_truth_test.csv')
            for i in gpt_models:
                run_experiment(GPTExperiment(i, get_prompt), "ground_truth_test.csv")
        except Exception:
            print(f'{attempts=}')
            attempts += 1
            # print(attempt)
            continue
    else:
        print(f'Max Attempts: {max_attempts}, reached!')
    # for i in gem_models:
        # run_experiment(GeminiExperiment(i, get_prompt), "ground_truth_test.csv")
    # # check_gpt('results/o4-mini/o4-mini_reasoning.csv')
    # print(get_average_vertex_difference([0, 0, 0, 0], [0, 0, 0, 0]))
    # print(get_average_vertex_difference([0, 0, 2, 2], [0, 0, 2, 2]))
    # print(get_average_vertex_difference([1, 0, 3, 2], [0, 0, 2, 2]))
    # print(get_average_vertex_difference([0, 0, 2, 2], [0, 2, 2, 4]))
    # print(get_average_vertex_difference([0, 0, 1, 1], [3, 4, 4, 5]))



    