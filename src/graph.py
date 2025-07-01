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
import json
from generate_ground_truth import tool_dict
matplotlib.use('AGG')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from pathlib import Path

            
def plot_numb_boxes_box_and_whiskers_comparison(csv_paths: list):
    one_box_data = []
    multi_box_data = []
    labels = [Path(path).stem for path in csv_paths]
    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
        one_box_ious = []
        multi_box_ious = []
        for index, row in df.iterrows():
            target_boxes = ast.literal_eval(row['target_bboxes'])
            iou_dict = ast.literal_eval(row['ious'])
            if len(target_boxes.keys()) == 1:
                for i in iou_dict.values():
                    one_box_ious.append(float(i))
            elif len(target_boxes.keys()) > 1:
                for i in iou_dict.values():
                    multi_box_ious.append(float(i))
            else:
                continue
        if one_box_ious:
            one_box_data.append(pd.DataFrame({'ious': one_box_ious, 'model': [label] * len(one_box_ious)}))
        if multi_box_ious:
            multi_box_data.append(pd.DataFrame({'ious': multi_box_ious, 'model': [label] * len(multi_box_ious)}))
    df_one = pd.concat(one_box_data, ignore_index=True) if one_box_data else pd.DataFrame()
    df_multi = pd.concat(multi_box_data, ignore_index=True) if multi_box_data else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(csv_paths) * 3), 8), sharey=True)
    if not df_one.empty:
        sns.boxplot(data=df_one, x='model', y='ious', ax=axes[0], palette='Set2', showfliers=False)
        sns.stripplot(data=df_one, x='model', y='ious', ax=axes[0], color='black', alpha=0.6, size=4, jitter=True)
        axes[0].set_title('One Grasp Prediction', fontsize=16)
        axes[0].set_xlabel('Model', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
    if not df_multi.empty:
        sns.boxplot(data=df_multi, x='model', y='ious', ax=axes[1], palette='Set2', showfliers=False)
        sns.stripplot(data=df_multi, x='model', y='ious', ax=axes[1], color='black', alpha=0.6, size=4, jitter=True)
        axes[1].set_title('Two Grasp Predictions', fontsize=16)
        axes[1].set_xlabel('Model', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
    for ax in axes:
        ax.set_ylabel('IoU', fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='y', labelsize=12)
    plt.suptitle('Grasp Prediction IoU Comparison by Amount of Grasp Predictions', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    fig.savefig('numb_boxes_comparison.png', dpi=300, bbox_inches='tight')

def plot_box_and_whiskers_comparison(csv_paths: list, labels=None):
    counter = 0
    all_data = []
    if labels is None:
        labels = [Path(path).stem for path in csv_paths]
    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
        ious = []
        for index, row in df.iterrows():
            iou_dict = ast.literal_eval(row['ious'])
            for i in iou_dict.values():
                assert isinstance(i, float)
                ious.append(float(i))
        model_data = {'ious': ious,'model': [label] * len(ious)}
        all_data.append(pd.DataFrame(model_data))
    combined_df = pd.concat(all_data, ignore_index=True)
    print(len(combined_df), len(combined_df['model'].unique()), len(combined_df['ious'].unique()))
    plt.figure(figsize=(max(12, len(csv_paths) * 3), 10))
    sns.boxplot(data=combined_df, x='model', y='ious', palette='Set2', showfliers=False)
    sns.stripplot(data=combined_df, x='model', y='ious', color='black', alpha=0.6, size=4, jitter=True)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel('Model', fontsize=16)
    plt.ylabel('IoU', fontsize=16)
    plt.title('Grasp Prediction Intersection over Union for All Models', fontsize=18)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()
    plt.savefig('results/all_model_comparison.png', dpi=300, bbox_inches='tight')
        
def plot_prediction_grid(csv_path, numb_of_imgs, gt_file='ground_truth_test.csv'):

    tools = []
    for key, value in tool_dict.items():
        name, _, _ = value
        tools.append(name)
    print(tools)
    tool_cycle = cycle(tools)
    print(next(tool_cycle))
    # tool_cycle = next(tool_cycle)
    df = pd.read_csv(csv_path, sep=';')
    df.columns = df.columns.str.replace('"', '', regex=False)
    gt = pd.read_csv(gt_file, sep=';')
    gt.columns = gt.columns.str.replace('"', '', regex=False)
    imgs = []
    #found = False
    #current_tool = next(tool_cycle)
    for i in range(numb_of_imgs):
        current_tool = next(tool_cycle)
        print(current_tool)
        for index, row in gt.iterrows():
            #print(current_tool)
            if row['tool'].strip() == current_tool.strip():
                if row['img_path'] not in imgs:
                    imgs.append(row['img_path'])
                else:
                    continue
                break

            else:
                continue
    print(imgs)
    for idx, i in enumerate(imgs):
        assert i not in imgs[(idx+1):]
    print(df.columns)
    # breakpoint()
    img_paths = df['img_path'].to_list()
    counter = 0
    idxs = []
    for i in imgs:
        for idx, j in enumerate(img_paths):
            if i.strip() == j.strip():
                idxs.append(idx)
    print(len(idxs))
    print(idxs)
    num_imgs = len(idxs)
    cols = int(math.sqrt(num_imgs))
    rows = math.ceil(num_imgs/cols)
    print(num_imgs, cols, rows)
    fig, axes = plt.subplots(rows, cols, figsize=(40, 7 * rows))
    axes = axes.flatten()
    for idx in range(numb_of_imgs):
        row = df.loc[idx]
        img = mpimg.imread(row['img_path'].strip())
        axes[idx].imshow(img)
        axes[idx].axis('off')
        pred_boxes = ast.literal_eval(row['pred_bboxes'])
        target_boxes = ast.literal_eval(row['target_bboxes'])
        if len(target_boxes) > 1:
            pred_x1, pred_y1, pred_x2, pred_y2 = (pred_boxes[list(pred_boxes.keys())[0]])
            try:
                pred2_x1, pred2_y1, pred2_x2, pred2_y2 = (pred_boxes[list(pred_boxes.keys())[1]])
            except IndexError:
                pred2_x1, pred2_y1, pred2_x2, pred2_y2 = (0, 0, 0, 0)
                #this ONLY happens because i initially forgot to add the second 'none_found' box to the pred_bboxes
                #where it assumbes that it is 0,0,0,0
            gt_x1, gt_y1, gt_x2, gt_y2 = [target_boxes[list(target_boxes.keys())[1]]['x_1'], target_boxes[list(target_boxes.keys())[1]]['y_1'],
                                          target_boxes[list(target_boxes.keys())[1]]['x_2'], target_boxes[list(target_boxes.keys())[1]]['y_2']]
            gt2_x1, gt2_y1, gt2_x2, gt2_y = [target_boxes[list(target_boxes.keys())[0]]['x_1'], target_boxes[list(target_boxes.keys())[0]]['y_1'],
                                          target_boxes[list(target_boxes.keys())[0]]['x_2'], target_boxes[list(target_boxes.keys())[0]]['y_2']]
            #results are structured {'index': 'thumb':}, but ground truth is structured {'thumb': 'index':}
            gt_x1 = gt_x1 * 1000
            gt_y1 = gt_y1 * 1000
            gt_x2 = gt_x2 * 1000
            gt_y2 = gt_y2 * 1000
            gt2_x1 = gt2_x1 * 1000
            gt2_y1 = gt2_y1 * 1000
            gt2_x2 = gt2_x2 * 1000
            gt2_y2 = gt2_y * 1000
            pred_x1 = pred_x1 * 1000
            pred_y1 = pred_y1 * 1000
            pred_x2 = pred_x2 * 1000
            pred_y2 = pred_y2 * 1000
            pred2_x1 = pred2_x1 * 1000
            pred2_y1 = pred2_y1 * 1000
            pred2_x2 = pred2_x2 * 1000
            pred2_y2 = pred2_y2 * 1000
            pred_rect_1 = patches.Rectangle((pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1, linewidth=3, edgecolor='red', facecolor='none')
            pred_rect_2 = patches.Rectangle((pred2_x1, pred2_y1), pred2_x2 - pred2_x1, pred2_y2 - pred2_y1, linewidth=3, edgecolor='red', facecolor='none')
            gt_rect_1 = patches.Rectangle((gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1, linewidth=3, edgecolor='lime', facecolor='none')
            gt_rect_2 = patches.Rectangle((gt2_x1, gt2_y1), gt2_x2 - gt2_x1, gt2_y2 - gt2_y1, linewidth=3, edgecolor='lime', facecolor='none')
            axes[idx].add_patch(pred_rect_1)
            axes[idx].add_patch(pred_rect_2)
            axes[idx].add_patch(gt_rect_1)
            axes[idx].add_patch(gt_rect_2)
            axes[idx].text(0.5, -0.1, os.path.basename(row['img_path']), transform=axes[idx].transAxes, ha='center', va='top', fontsize=10)
            axes[idx].text(0.5, -0.18, f"IoU: {row['ious']}", transform=axes[idx].transAxes, ha='center', va='top', fontsize=10)
            if 'index' in ast.literal_eval(row['pred_bboxes']).keys():
                axes[idx].text(pred_x1, pred_y1 - 0.04,'Pred: index',color='red',fontsize=7,backgroundcolor='white')
                axes[idx].text(pred2_x1, pred2_y1 - 0.04,'Pred: thumb',color='red',fontsize=7,backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04,'GT: index',color='lime',fontsize=7,backgroundcolor='white')
                axes[idx].text(gt2_x1, gt2_y1 - 0.04,'GT: thumb',color='lime',fontsize=7,backgroundcolor='white')
            elif 'hand1' in ast.literal_eval(row['pred_bboxes']).keys():
                axes[idx].text(pred_x1, pred_y1 - 0.04,'Pred: hand1',color='red',fontsize=7,backgroundcolor='white')
                axes[idx].text(pred2_x1, pred2_y1 - 0.04,'Pred: hand2',color='red',fontsize=7,backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04,'GT: hand1',color='lime',fontsize=7,backgroundcolor='white')
                axes[idx].text(gt2_x1, gt2_y1 - 0.04,'GT: hand2',color='lime',fontsize=7,backgroundcolor='white')
            elif 'none_found' in ast.literal_eval(row['pred_bboxes']).keys():
                continue
            else:
                print(ast.literal_eval(row['pred_bboxes']).keys())
                assert 0 > 1
        elif len(target_boxes) == 1:
            try:
                pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[list(pred_boxes.keys())[0]]
            except ValueError:
                pred_x1, pred_y1, pred_x2, pred_y2 = (0, 0, 0, 0)
                #this ONLY happens because i initially forgot to add the second 'none_found' box to the pred_bboxes
                #where it assumbes that it is 0,0,0,0.
                #it also only happens if the model produces either no json or a bad json
            print(type(pred_x1), type(pred_y1), type(pred_x2), type(pred_y2))
            gt_x1, gt_y1, gt_x2, gt_y2 = [target_boxes[list(target_boxes.keys())[0]]['x_1'], target_boxes[list(target_boxes.keys())[0]]['y_1'], 
                                          target_boxes[list(target_boxes.keys())[0]]['x_2'], target_boxes[list(target_boxes.keys())[0]]['y_2']]
            print(type(target_boxes[list(target_boxes.keys())[0]]))
            print(type(gt_x1), type(gt_y1), type(gt_x2), type(gt_y2))
            print(gt_x1)
            gt_x1 = gt_x1 * 1000
            gt_y1 = gt_y1 * 1000
            gt_x2 = gt_x2 * 1000
            gt_y2 = gt_y2 * 1000
            pred_x1 = pred_x1 * 1000
            pred_y1 = pred_y1 * 1000
            pred_x2 = pred_x2 * 1000
            pred_y2 = pred_y2 * 1000
            pred_rect = patches.Rectangle((pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1, linewidth=3, edgecolor='red', facecolor='none')
            gt_rect = patches.Rectangle((gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1, linewidth=3, edgecolor='lime', facecolor='none')
            axes[idx].add_patch(pred_rect)
            axes[idx].add_patch(gt_rect)
            axes[idx].text(0.5, -0.1, os.path.basename(row['img_path']), transform=axes[idx].transAxes, ha='center', va='top', fontsize=10)
            axes[idx].text(0.5, -0.18, f"IoU: {row['ious']}", transform=axes[idx].transAxes, ha='center', va='top', fontsize=10)
            if 'hand' in ast.literal_eval(row['pred_bboxes']).keys():
                axes[idx].text(pred_x1, pred_y1 - 0.02,'Pred: hand',color='red',fontsize=7,backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04,'GT: hand',color='lime',fontsize=7,backgroundcolor='white')
            elif 'index' in ast.literal_eval(row['pred_bboxes']).keys():
                axes[idx].text(pred_x1, pred_y1 - 0.04,'Pred: index',color='red',fontsize=7,backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04,'GT: index',color='lime',fontsize=7,backgroundcolor='white')
            elif 'none_found' in ast.literal_eval(row['pred_bboxes']).keys():
                continue
            else:
                print(ast.literal_eval(row['pred_bboxes']).keys())
                assert 0 > 1
        else:

            assert 0 > 1
        for j in range(idx + 1, len(axes)):
            axes[j].axis('off') 
    model_name = Path(csv_path).stem
    plt.suptitle(f'Grasp Prediction Grid for {model_name}', fontsize=45, y= 0.99)
    # plt.legend(f'Red = {model_name}, Green = Groun  d Truth')
    plt.tight_layout()
    red_patch = patches.Patch(color='red', label=f'Red = {model_name}')
    green_patch = patches.Patch(color='lime', label='Green = Ground Truth')
    fig.legend(handles=[red_patch, green_patch], loc='upper right', fontsize=25)
    plt.show()
    plt.savefig(f'results/{model_name}_prediction_grid.png')

            

            



        

  

if __name__ == "__main__":
    # model_list = ['results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307.csv', 'results/gemini-2.5-flash-lite-preview-06-17.csv',
    #                                   'results/gemini-2.5-flash.csv', 'results/gemini-2.0-flash-lite.csv', 'results/gpt-4.1-mini.csv', 'results/gpt-4.1-nano.csv',
    #                                   'results/grok-2-vision-1212.csv', 'results/o4-mini.csv', 'results/owl_vit_prompt_1.csv', 'results/yolo_uniow_prompt_1.csv', 'results/yolo_world_prompt_1.csv']
    # for i in model_list:
    #     plot_prediction_grid(i, 64)
    # plot_box_and_whiskers_comparison(model_list)
    plot_prediction_grid('results/yolo_uniow.csv', 64)
