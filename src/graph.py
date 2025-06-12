#graphing functions, redo box and whiskers, add graphs by image size maybe?
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import regex as re
import math
from itertools import cycle
import ast
import os
import numpy as np
from pathlib import Path
matplotlib.use('AGG')
tool_dict = {'drill' : 'drill', 'wacker' : 'weed_wacker', 'glue' : 'glue_gun', 'saw' : 'circular_saw', 'nail' : 'nail_gun', 
    'screwdriver' : 'screwdriver', 'wrench' : 'wrench', 'solder' : 'solder_iron', 'allen' : 'allen_key', 'hammer' : 'hammer'}
models_regex = r"claude-3-5-haiku-latest|claude-3-haiku-20240307|gemini-1.5-flash|gemini-2.0-flash-lite|gemini-2.0-flash|gemini-2.5-flash-preview-05-20|grok-2-vision-1212|gpt-4.1-mini|gpt-4.1-nano|gpt-4o-mini|o4-mini"
tool_remove_regex = r'screwdriver|glue gun'
def check(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
    for i in df['iou'].values.tolist():
        assert isinstance(i, float)
        print(i, type(i))
        # print(type())



def plot_box_and_whiskers(iou_dict: dict = None):
    if iou_dict is None:
        assert 0 > 1
    df = pd.DataFrame(iou_dict)
    breakpoint()
    print(df.columns.tolist())
    print(f'{len(df.columns.tolist())}=')
    # breakpoint()
    # df.to_csv('test.csv', sep=';', encoding='utf-8')
    # breakpoint()
    print(df)
    # breakpoint()
    print(df)
    df = df.melt(var_name="Noun", value_name="IoU")
    plt.figure(figsize=(40, 20))
    sns.boxplot(x='Noun', y='IoU', data=df, palette='Set2', showfliers=False)
    # sns.stripplot(x='Noun', y='IoU', data=df, color='black', alpha=0.7, size=6, jitter=True)
    # plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel('Iou', fontsize=30)
    plt.ylabel('Model', fontsize=30)
    plt.title('Boxplot of IoUs for All Model Performance on My Dataset', fontsize=50)
    plt.tight_layout()
    plt.show()
    plt.savefig('src/best_all_mocdel.png')

def plot_prediction_grid(csv_path, numb_of_imgs, gt_file='src/ground_truth.csv'):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    gt = pd.read_csv(gt_file, sep=';', encoding='utf-8')
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
    gt.columns = gt.columns.str.replace('"', '', regex=False)
    img_paths_by_tool = get_img_paths_by_tool(gt_file)
    amount_per_tool = numb_of_imgs // 10
    #amount of images in the prediction tool that are guaranteed per tool
    left_over = numb_of_imgs - amount_per_tool * 10
    #left over amount of images to make the grid a square
    print(img_paths_by_tool)
    print(amount_per_tool)
    key_cycle = cycle(img_paths_by_tool.keys())
    print(key_cycle)
    imgs = []
    counter = 0
    index = 0
    keys = list(img_paths_by_tool.keys())
    while counter < numb_of_imgs:
        key = next(key_cycle)
        path = img_paths_by_tool[key][index]
        if key == keys[-1]:
            index += 1
        imgs.append(path)
        counter += 1
    print(imgs)
    print(len((imgs)))
    for idx, i in enumerate(imgs):
        if i in imgs[idx+1:]:
            print(i)
            breakpoint()
        else:
            pass
    num_imgs = len(imgs)
    cols = int(math.sqrt(num_imgs))
    rows = math.ceil(num_imgs/cols)
    print(num_imgs, cols, rows)
    fig, axes = plt.subplots(rows, cols, figsize=(40, 7 * rows))
    axes = axes.flatten()
    for idx, img_path in enumerate(imgs):
        img = mpimg.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')  # Turn off axes
        to_check_row_gt = gt[gt['img_path'].astype(str) == str(img_path)]
        img_id = to_check_row_gt['img_id']
        to_check_row_df = df[df['img_id'].astype(str) == str(img_id.iloc[0])]
        iou = to_check_row_df['iou']
        gt_bbox = ast.literal_eval(to_check_row_gt['bbox'].iloc[0])
        gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bbox
        gt_width = gt_x_max - gt_x_min
        gt_height = gt_y_max - gt_y_min
        gt_rect = patches.Rectangle((gt_x_min, gt_y_min), gt_width, gt_height, linewidth=3, edgecolor='lime', facecolor='none', label='Green = Ground Truth')
        pred_bbox = ast.literal_eval(to_check_row_df['pred_bbox'].iloc[0])
        if len(pred_bbox) != 4:
            pred_bbox = [0, 0, 0, 0]
        print(f"type: {type(pred_bbox)}")
        print(f"bbox: {pred_bbox}")
        df_x_min, df_y_min, df_x_max, df_y_max = pred_bbox
        df_width = df_x_max - df_x_min
        df_height = df_y_max - df_y_min
        model_match = re.search(models_regex, csv_path)
        # if model_match:
        #     if 'reason' in csv_path:
        #         model_name = f'{model_match.group(0)}_w_reasoning'
        #     else:
        #         model_name = model_match.group(0)
        model_name = 'YOLO World'
        print(model_name)
        df_rect = patches.Rectangle((df_x_min, df_y_min), df_width, df_height, linewidth=3, edgecolor='red', facecolor='none', label=f'Red = {model_name}')
        axes[idx].add_patch(gt_rect)
        axes[idx].add_patch(df_rect)
        axes[idx].text(0.5, -0.1, f"{os.path.basename(to_check_row_gt['img_path'].iloc[0])}", transform=axes[idx].transAxes,
            ha='center', va='top', fontsize=11, rotation=0)
        # axes[idx].set_title(f"{to_check_row_gt['img_path'].iloc[0]}", fontsize=5)
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'Prediction Grid for {model_name}', fontsize=45, y= 0.99)
    plt.legend(f'Red = {model_name}, Green = Ground Truth')
    plt.tight_layout()
    red_patch = patches.Patch(color='red', label=f'Red = {model_name}')
    green_patch = patches.Patch(color='lime', label='Green = Ground Truth')
    fig.legend(handles=[red_patch, green_patch], loc='upper right', fontsize=25)
    plt.show()
    plt.savefig(f'results/{model_name}_prediction_grid.png')
def aggregate_data(root_dir = 'results'):
    iou_dict = {}
    vlm_csvs = []
    owl_yolo_csvs = []
    for i in Path(root_dir).rglob('*.csv'):
        if 'owl' in str(i) or 'yolo' in str(i):
            if str(i.parent) == 'results':
                owl_yolo_csvs.append(str(i))
            else:
                continue
        else:
            vlm_csvs.append(str(i))
    # for i in owl_yolo_csvs:
    print(len(owl_yolo_csvs))
    for idx, i in enumerate(owl_yolo_csvs):
        print(idx)
        print(i)
        assert i not in (owl_yolo_csvs[idx + 1:])
    breakpoint()
    for idx, i in enumerate(owl_yolo_csvs):
        
        result = get_iou_single(i, idx)
        if result is None:
            continue
        else:
            result = result[0]
        hand_prompt, index_prompt = result
        print(hand_prompt)
        print(index_prompt)
        print(len(hand_prompt))
        print(len(index_prompt))
        hand_prompt_values = list(hand_prompt.values())[0]
        index_prompt_values = list(index_prompt.values())[0]
        print(hand_prompt_values)
        print(index_prompt_values)
        print(f"{len(hand_prompt_values)=}")
        print(f"{len(index_prompt_values)=}")
        conjoined_values = hand_prompt_values + index_prompt_values
        hand_prompt_key = str(list(hand_prompt.keys())[0])
        index_prompt_key = str(list(index_prompt.keys())[0])
        conjoined_key = f'{hand_prompt_key}-{index_prompt_key}'
        key = re.search(r'world|owl|uniow', conjoined_key)
        if key:
            if key.group(0) == 'world':
                key = 'YOLO-World'
            elif key.group(0) == 'owl':
                key = 'Owl-VIT'
            else:
                key = 'YOLO-UniOW'
        iou_dict[key] = conjoined_values
    
    filtered = []
    for i in vlm_csvs:
        if 'reason' not in i and '_reason' not in i:
            filtered.append(i)
    vlm_csvs = filtered
    print(vlm_csvs)
    breakpoint()
    vlm_ious = get_ious(vlm_csvs)
    for key, value in vlm_ious.items():
        iou_dict[key] = value
    iou_dict = dict(sorted(iou_dict.items()))
    return iou_dict

def get_iou_single(path, idx):
    #works on owl and yolo
    combined = []
    ground_truth = pd.read_csv('src/ground_truth_owl.csv', delimiter=';', encoding='utf-8')
    ground_truth.columns = ground_truth.columns.str.replace('"', '', regex=False)
    hand_regex = r'grab|grip|grasp|bar|hold|handle'
    finger_regex = r'button|lever|switch|press|toggle|trigger'
    hand = ['grab', 'grip', 'grasp', 'bar', 'hold', 'handle']
    finger = ['button', 'lever', 'switch', 'press', 'toggle', 'trigger']
    hand_ious = []
    finger_ious = []
    df = pd.read_csv(path, delimiter=';', encoding='utf-8')
    df.columns = df.columns.str.replace('"', '', regex=False)
    for index, row in df.iterrows():
        to_check_row = ground_truth[ground_truth['img_id'] == row['img_id']]
        if 'index' in to_check_row['annotation_type'].iloc[0]:
            finger_ious.append(float(row['iou']))
        elif 'four' in to_check_row['annotation_type'].iloc[0]:
            hand_ious.append(float(row['iou']))
        else:
            print(f"Nu bueno")
            print(to_check_row)
            breakpoint()
    to_check_row_hand = df[df['img_id'] == 0]
    to_check_row_index = df[df['img_id'] == 1]
    suffix = ''
    if 'yolo' in path:
        if 'world' in path:
            suffix += 'yolo_world'
            if '_s_' in path:
                return None
                # suffix += '_small'
            elif '_m_' in path:
                return None
                # suffix += '_medium'
            elif '_l_' in path:
                suffix += '_large_'
            if '1280' in path:
                suffix += '1280ft'
            else:
                return None
                # suffix += '640ft'
                
        elif 'uniow' in path:
            suffix += 'uniow'
            if '_s_' in path:
                return None
                # suffix += '_small'
            elif '_m_' in path:
                return None
                # continue
                # suffix += '_medium'
            elif '_l_' in path:
                suffix += '_large_'
    elif 'vit' in path:
        suffix += '_owl_vit'
    else:
        print('yikes')
        breakpoint()
    prompt_hand = str((to_check_row_hand['noun'].iloc[0]).replace(']', '').replace('[', '').replace("'", ''))
    prompt_index = str((to_check_row_index['noun'].iloc[0]).replace(']', '').replace('[', '').replace("'", ''))
    prompt_hand = re.sub(tool_remove_regex, '', str(prompt_hand))
    prompt_index = re.sub(tool_remove_regex, '', str(prompt_index))
    if 'area to grab safely' not in prompt_hand:
        return None
    combined.append(({f'{prompt_hand}{suffix}' : hand_ious}, {f'{prompt_index}{suffix}' : finger_ious}))
    return combined
    
def get_ious(csv_list):
    ious = {}
    test_list = []
    for path in csv_list:
        iou_list = []
        df = pd.read_csv(path, delimiter=';', encoding='utf-8')
        # I decided to change the delimiter to semicolon as none of the vlm responses used it 
        # and it looks a little bit cleaner in my opinion.
        df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
        print("Columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        model_match = re.search(models_regex, path)
        if model_match is None:
            print(f"NONE MODEL MATCH: {path}")
            breakpoint()
        for index, row in df.iterrows():
            iou_list.append(row['iou'])
        if 'reason' in path:
            model_name = f"{model_match.group(0)}_reasoning"
        else:
            model_name = model_match.group(0)
        test_list.append(model_match)
        for idx, i in enumerate(test_list):
            assert i not in test_list[idx + 1:]
        ious[model_name] = iou_list
    return ious

def get_img_paths_by_tool(csv_path):
    img_path_dict = {}
    df = pd.read_csv(csv_path, sep=';')
    df.columns = df.columns.str.replace('"', '', regex=False)
    for i in tool_dict.keys():
        path_names = [j for j in df['img_path'] if i in j]
        img_path_dict[i] = path_names
    print(img_path_dict)
    return img_path_dict







if __name__ == "__main__":
    # check('results/grok-2-vision-1212.csv')
    plot_box_and_whiskers(aggregate_data())
    # plot_prediction_grid('results/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival_2.csv', 64)






