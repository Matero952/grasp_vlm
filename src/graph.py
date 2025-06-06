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
models_regex = r"claude-3-5-haiku-latest|claude-3-haiku-20240307|gemini-1.5-flash|gemini-2.0-flash|gemini-2.0-flash-lite|gemini-2.5-flash-preview-05-20|grok-2-vision-1212|gpt-4.1-mini|gpt-4.1-nano|gpt-4o-mini|o4-mini"

def plot_box_and_whiskers(iou_dict: dict):
    df = pd.DataFrame(iou_dict)
    print(df)
    breakpoint()
    print(df)
    df = df.melt(var_name="Noun", value_name="IoU")
    plt.figure(figsize=(30, 25))
    sns.boxplot(x = 'IoU', y = 'Noun', data=df)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Iou', fontsize=20)
    plt.ylabel('Noun', fontsize=20)
    plt.title('Boxplot of IoUs for Owl and YOLO', fontsize=25)
    plt.tight_layout()
    plt.show()
    plt.savefig('src/iou_boxplt_t.png')

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
        if model_match:
            if 'reason' in csv_path:
                model_name = f'{model_match.group(0)}_w_reasoning'
            else:
                model_name = model_match.group(0)
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

def get_owl_single(gt_path):
    owl_yolo_csv_list = []
    # rest_list = []
    ious = {}
    gt = pd.read_csv(gt_path, delimiter=';', encoding='utf-8')
    gt.columns = gt.columns.str.replace('"', '', regex=False)
    hand_regex = r'grab|grip|grasp|bar|hold|handle'
    finger_regex = r'button|lever|switch|press|toggle|trigger'
    hand = ['grab', 'grip', 'grasp', 'bar', 'hold', 'handle']
    finger = ['button', 'lever', 'switch', 'press', 'toggle', 'trigger']
    def_path = Path('results')
    for i in def_path.rglob('*.csv'):
        # print(str(i))
        if 'owl' in str(i) or 'yolo' in str(i):
            # print(str(i))
            owl_yolo_csv_list.append(str(i))
        else:
            # rest_list.append(str(i))
            continue
    # print(csv_list)
    for path in owl_yolo_csv_list:
        hand_ious = []
        finger_ious = []
        df = pd.read_csv(path, delimiter=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False)
        # print(df.columns)
        # for col in df.columns:
        #     print(f'{repr(col)=}')
        for index, row in df.iterrows():
            to_check_row = gt[gt['img_id'] == row['img_id']]
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
        # print(to_check_row)
        if 'yolo' in path:
            if '_l_' in path:
                suffix = '_yolo_large'
            elif '_m_' in path:
                suffix = '_yolo_medium'
            else:
                suffix = '_yolo_small'
        else:
            suffix = ''
        to_check_row = gt[gt['img_id'] == row['img_id']]
        prompt_hand = (to_check_row_hand['noun'].iloc[0]).replace(']', '').replace('[', '').replace("'", '')
        prompt_index = (to_check_row_index['noun'].iloc[0]).replace(']', '').replace('[', '').replace("'", '')
        ious[f'{prompt_hand}{suffix}'] = hand_ious
        ious[f'{prompt_index}{suffix}'] = finger_ious
        
        print(f"Max for {prompt_hand}{suffix}: {max(hand_ious)}. Image: {np.argmax(hand_ious)}") 
        print(f"Max for {prompt_index}{suffix}: {max(finger_ious)} Image: {np.argmax(finger_ious)}")
    breakpoint()
    return ious


def get_ious(csv_list):
    ious = {}
    for path in csv_list:
        iou_list = []
        df = pd.read_csv(path, delimiter=';', encoding='utf-8')
        # I decided to change the delimiter to semicolon as none of the vlm responses used it 
        # and it looks a little bit cleaner in my opinion.
        df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
        print("Columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        model_match = re.search(models_regex, path)
        for index, row in df.iterrows():
            iou_list.append(row['iou'])
        if 'reason' in path:
            model_name = f"{model_match.group(0)}_reasoning"
        else:
            model_name = model_match.group(0)
        ious[model_name] = iou_list
    
    print(ious)
    # print(len(gg))
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
    # plot_box_and_whiskers(get_owl_single('src/ground_truth_owl.csv'))
    # print(len(['results/claude-3-5-haiku-latest-reasoning_reason.csv', 'results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307-reasoning_reason.csv', 
    #                 'results/claude-3-haiku-20240307.csv', 'results/gemini-1.5-flash-reasoning.csv', 'results/gemini-1.5-flash.csv', 'results/gemini-2.0-flash-lite-reasoning.csv',
    #                 'results/gemini-2.0-flash-lite.csv', 'results/gemini-2.0-flash-reasoning_reason.csv', 'results/gemini-2.0-flash.csv', 'results/gemini-2.5-flash-preview-05-20-reasoning.csv',
    #                 'results/gemini-2.5-flash-preview-05-20.csv', 'results/grok-2-vision-1212-reasoning_reason.csv', 'results/grok-2-vision-1212.csv', 'results/gpt-4.1-mini_reasoning.csv',
    #                 'results/gpt-4.1-mini.csv', 'results/gpt-4.1-nano_reasoning.csv', 'results/gpt-4.1-nano.csv', 'results/gpt-4o-mini_reasoning.csv', 'results/gpt-4o-mini.csv',
    #                 'results/o4-mini_reasoning.csv', 'results/o4-mini.csv'
    #                 ]))
    # breakpoint()
    # plot_box_and_whiskers(get_ious(['results/claude-3-5-haiku-latest-reasoning_reason.csv', 'results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307-reasoning_reason.csv', 
    #                 'results/claude-3-haiku-20240307.csv', 'results/gemini-1.5-flash-reasoning.csv', 'results/gemini-1.5-flash.csv', 'results/gemini-2.0-flash-lite-reasoning.csv',
    #                 'results/gemini-2.0-flash-lite.csv', 'results/gemini-2.0-flash-reasoning_reason.csv', 'results/gemini-2.0-flash.csv', 'results/gemini-2.5-flash-preview-05-20-reasoning.csv',
    #                 'results/gemini-2.5-flash-preview-05-20.csv', 'results/grok-2-vision-1212-reasoning_reason.csv', 'results/grok-2-vision-1212.csv', 'results/gpt-4.1-mini_reasoning.csv',
    #                 'results/gpt-4.1-mini.csv', 'results/gpt-4.1-nano_reasoning.csv', 'results/gpt-4.1-nano.csv', 'results/gpt-4o-mini_reasoning.csv', 'results/gpt-4o-mini.csv',
    #                 'results/o4-mini_reasoning.csv', 'results/o4-mini.csv'
    #                 ]))
    plot_prediction_grid('results/o4-mini.csv', 64)
    # get_img_paths_by_tool('src/ground_truth.csv')





