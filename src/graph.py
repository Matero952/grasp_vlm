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
import csv
#TODO make a prediction grid for the really bad images, take the best prediction for that image and plot it

def plot_dataset_info_k_means_cluster(model_list):
    data = []
    pred_data = []
    #i want a list of lists that reprersent average gtbbox width, average gtbbox height
    with open('ground_truth_test.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            gts = ast.literal_eval(row['bboxes'])
            for val in gts.values():
                x1, y1, x2, y2 = [val[i] for i in val]
                width = (x2 - x1) * 1000
                height = (y2 - y1) * 1000
                print(width, height)
                data.append([width, height])
    for path in model_list:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                preds = ast.literal_eval(row['pred_bboxes'])
                for val in preds.values():
                    try:
                        x1, y1, x2, y2 = val
                    except ValueError:
                        print(val)
                        print(row)
                        print(path)
                    # breakpoint()
                    width = (x2 - x1) * 1000
                    height = (y2 - y1) * 1000
                    pred_data.append([width, height])
        # Get the cluster centroids and labels

    gt_widths, gt_heights = zip(*data)
    pred_widths, pred_heights = zip(*pred_data)
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pred_widths, pred_heights, color='blue', label='Predicted', s=0.5)
    plt.scatter(gt_widths, gt_heights, color='red', label='Ground Truth', s=0.5)
    # plt.scatter(pred_widths, pred_heights, color='blue', label='Predicted', s=0.5)

    # Add labels and legend
    plt.xlabel('Bounding Box Width')
    plt.ylabel('Bounding Box Height')
    plt.title('Ground Truth vs. Predicted Bounding Box Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('tesss.png')
    # Plot the points with cluster coloring



# def plot_bad_prediction_grid(csv_list):
#     #lets plot maybe the 16 worst predictions in the entire 
def plot_ious_by_img(csv_paths: list[str]):
    counter = 0
    img_iou_dict = {}
    for i in range(500):
        img_iou_dict[i] = []
    for path in csv_paths:
        df = pd.read_csv(path, sep=';')
        df.columns = df.columns.str.replace('"', '', regex=False)
        for index, row in df.iterrows():
            img_id = int(row['img_id'])
            iou_values = [val for val in ast.literal_eval(row['ious']).values()]
            for val in iou_values:
                img_iou_dict[img_id].append(val)
                counter += 1
    print(counter)
    breakpoint()
    avg_img_iou = {}
    bad_img_ids = []
    all_img_ids = []
    for i in range(500):
        ious = img_iou_dict[i]
        average_iou = sum(ious) / len(ious)
        avg_img_iou[i] = average_iou
        all_img_ids.append((i, average_iou))
    all_img_ids.sort(key=lambda x: x[1], reverse=True)
    print(all_img_ids)
    return all_img_ids
    #     if average_iou < 0.01:
    #         bad_img_ids.append(i)
    # print(avg_img_iou)
    # df = pd.DataFrame.from_dict(avg_img_iou, orient='index').stack().reset_index()
    # df.columns = ['img_id', 'index', 'iou']
    # df = df.drop('index', axis=1)
    # print(df)
    # plt.figure(figsize=(10, 6))
    # plt.bar(df['img_id'], df['iou'], color='blue', alpha=0.7)
    # plt.xlabel('Image ID')
    # plt.ylabel('IoU')
    # plt.title('Average IoU by Image ID for All Models')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('results/ious_by_img.png')
    # print(bad_img_ids)
    # class_by_bad_id = {}
    # with open('ground_truth_test.csv', 'r') as f:
    #     reader = csv.DictReader(f, delimiter=';')
    #     for i in bad_img_ids:
    #         f.seek(0)  # Reset to beginning of file
    #         next(reader)  # Skip header row after reset
    #         for j in reader:
    #             if str(j['img_id']).strip() == str(i):

    #                 if j['tool'] not in class_by_bad_id:
    #                     class_by_bad_id[j['tool']] = 1
    #                 else:
    #                     class_by_bad_id[j['tool']] += 1
    # print(class_by_bad_id)
    # df = pd.DataFrame.from_dict(class_by_bad_id, orient='index', columns=['value'])
    # df.index.name = 'key'
    # df = df.reset_index()
    # df.plot(x='key', y='value', kind='bar', figsize=(8, 12))
    # plt.title('Occurences for Average iou < 0.01 By Class Across All Models')
    # plt.show()
    # plt.savefig('test.png')

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
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
    if not df_one.empty:
        sns.boxplot(data=df_one, x='model', y='ious', ax=axes[0], palette='Set2', showfliers=False)
        # sns.stripplot(data=df_one, x='model', y='ious', ax=axes[0], color='black', alpha=0.6, size=4, jitter=True)
        axes[0].set_title('One Grasp Prediction', fontsize=16)
        axes[0].set_xlabel('Model', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
    if not df_multi.empty:
        sns.boxplot(data=df_multi, x='model', y='ious', ax=axes[1], palette='Set2', showfliers=False)
        # sns.stripplot(data=df_multi, x='model', y='ious', ax=axes[1], color='black', alpha=0.6, size=4, jitter=True)
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
    fig.savefig('results/numb_boxes_comparison.png', dpi=300, bbox_inches='tight')

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
                # assert isinstance(i, float)
                ious.append(float(i))
        model_data = {'ious': ious,'model': [label] * len(ious)}
        all_data.append(pd.DataFrame(model_data))
    combined_df = pd.concat(all_data, ignore_index=True)
    print(len(combined_df), len(combined_df['model'].unique()), len(combined_df['ious'].unique()))
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=combined_df, x='model', y='ious', palette='Set2', showfliers=False)
    # sns.stripplot(data=combined_df, x='model', y='ious', color='black', alpha=0.6, size=4, jitter=True)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Model', fontsize=16)
    plt.ylabel('IoU', fontsize=16)
    plt.title('Grasp Prediction Intersection over Union for All Models', fontsize=18)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()
    plt.savefig('results/all_model_comparison.png', dpi=300, bbox_inches='tight')

def plot_predition_grid(model_img_tuples, gt_file='ground_truth_test.csv'):
    """
    Plot prediction grid for specific model-image combinations.
    
    Args:
        model_img_tuples: List of tuples (model_name, img_id) where:
                         - model_name: name of the model (used to find CSV file)
                         - img_id: specific image ID to look for in the CSV
        gt_file: path to ground truth CSV file
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    import ast
    import os
    import math
    from pathlib import Path
    
    # Load ground truth data
    gt = pd.read_csv(gt_file, sep=';')
    gt.columns = gt.columns.str.replace('"', '', regex=False)
    
    # Collect data for each model-image pair
    plot_data = []
    
    for model_name, img_id in model_img_tuples:
        # Construct CSV path for this model
        csv_path = f"results/{model_name}.csv"  # Adjust path format as needed
        try:
            # Load model predictions CSV
            df = pd.read_csv(csv_path, sep=';')
            df.columns = df.columns.str.replace('"', '', regex=False)
            # Find the row with matching img_id
            # Assuming img_id could be in 'img_path' column or a separate 'img_id' column
            if 'img_id' in df.columns:
                matching_row = df[df['img_id'] == img_id]
            else:
                # Look for img_id in img_path (basename without extension)
                matching_row = df[df['img_path'].apply(lambda x: Path(x).stem == str(img_id))]
            
            if not matching_row.empty:
                row_data = matching_row.iloc[0]
                plot_data.append({
                    'model_name': model_name,
                    'img_id': img_id,
                    'row': row_data
                })
            else:
                print(f"Warning: Image ID {img_id} not found in {csv_path}")
                
        except FileNotFoundError:
            print(f"Warning: CSV file {csv_path} not found for model {model_name}")
        except Exception as e:
            print(f"Error processing {model_name}, {img_id}: {e}")
    
    if not plot_data:
        print("No valid data found to plot")
        return
    
    # Calculate grid dimensions
    num_imgs = len(plot_data)
    cols = int(math.sqrt(num_imgs))
    rows = math.ceil(num_imgs / cols)
    
    print(f"Plotting {num_imgs} images in {rows}x{cols} grid")
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(40, 7 * rows))
    if num_imgs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each image
    for idx, data in enumerate(plot_data):
        model_name = data['model_name']
        img_id = data['img_id']
        row = data['row']
        
        # Load and display image
        img = mpimg.imread(row['img_path'].strip())
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Parse bounding boxes
        pred_boxes = ast.literal_eval(row['pred_bboxes'])
        target_boxes = ast.literal_eval(row['target_bboxes'])
        
        if len(target_boxes) > 1:
            # Handle case with multiple target boxes
            pred_x1, pred_y1, pred_x2, pred_y2 = (pred_boxes[list(pred_boxes.keys())[0]])
            try:
                pred2_x1, pred2_y1, pred2_x2, pred2_y2 = (pred_boxes[list(pred_boxes.keys())[1]])
            except (IndexError, KeyError):
                pred2_x1, pred2_y1, pred2_x2, pred2_y2 = (0, 0, 0, 0)
            
            gt_x1, gt_y1, gt_x2, gt_y2 = [target_boxes[list(target_boxes.keys())[1]]['x_1'], 
                                          target_boxes[list(target_boxes.keys())[1]]['y_1'],
                                          target_boxes[list(target_boxes.keys())[1]]['x_2'], 
                                          target_boxes[list(target_boxes.keys())[1]]['y_2']]
            gt2_x1, gt2_y1, gt2_x2, gt2_y2 = [target_boxes[list(target_boxes.keys())[0]]['x_1'], 
                                              target_boxes[list(target_boxes.keys())[0]]['y_1'],
                                              target_boxes[list(target_boxes.keys())[0]]['x_2'], 
                                              target_boxes[list(target_boxes.keys())[0]]['y_2']]
            
            # Scale coordinates
            scale_factor = 1000
            coords = [gt_x1, gt_y1, gt_x2, gt_y2, gt2_x1, gt2_y1, gt2_x2, gt2_y2,
                     pred_x1, pred_y1, pred_x2, pred_y2, pred2_x1, pred2_y1, pred2_x2, pred2_y2]
            coords = [c * scale_factor for c in coords]
            (gt_x1, gt_y1, gt_x2, gt_y2, gt2_x1, gt2_y1, gt2_x2, gt2_y2,
             pred_x1, pred_y1, pred_x2, pred_y2, pred2_x1, pred2_y1, pred2_x2, pred2_y2) = coords
            
            # Create rectangles
            pred_rect_1 = patches.Rectangle((pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1, 
                                          linewidth=3, edgecolor='red', facecolor='none')
            pred_rect_2 = patches.Rectangle((pred2_x1, pred2_y1), pred2_x2 - pred2_x1, pred2_y2 - pred2_y1, 
                                          linewidth=3, edgecolor='red', facecolor='none')
            gt_rect_1 = patches.Rectangle((gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1, 
                                        linewidth=3, edgecolor='lime', facecolor='none')
            gt_rect_2 = patches.Rectangle((gt2_x1, gt2_y1), gt2_x2 - gt2_x1, gt2_y2 - gt2_y1, 
                                        linewidth=3, edgecolor='lime', facecolor='none')
            
            # Add rectangles to plot
            axes[idx].add_patch(pred_rect_1)
            axes[idx].add_patch(pred_rect_2)
            axes[idx].add_patch(gt_rect_1)
            axes[idx].add_patch(gt_rect_2)
            
            # Add labels
            pred_keys = list(pred_boxes.keys())
            if 'index' in pred_keys:
                axes[idx].text(pred_x1, pred_y1 - 0.04, 'Pred: index', color='red', fontsize=7, backgroundcolor='white')
                axes[idx].text(pred2_x1, pred2_y1 - 0.04, 'Pred: thumb', color='red', fontsize=7, backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04, 'GT: index', color='lime', fontsize=7, backgroundcolor='white')
                axes[idx].text(gt2_x1, gt2_y1 - 0.04, 'GT: thumb', color='lime', fontsize=7, backgroundcolor='white')
            elif 'hand1' in pred_keys:
                axes[idx].text(pred_x1, pred_y1 - 0.04, 'Pred: hand1', color='red', fontsize=7, backgroundcolor='white')
                axes[idx].text(pred2_x1, pred2_y1 - 0.04, 'Pred: hand2', color='red', fontsize=7, backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04, 'GT: hand1', color='lime', fontsize=7, backgroundcolor='white')
                axes[idx].text(gt2_x1, gt2_y1 - 0.04, 'GT: hand2', color='lime', fontsize=7, backgroundcolor='white')
        
        elif len(target_boxes) == 1:
            # Handle case with single target box
            try:
                pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[list(pred_boxes.keys())[0]]
            except (ValueError, KeyError):
                pred_x1, pred_y1, pred_x2, pred_y2 = (0, 0, 0, 0)
            
            gt_x1, gt_y1, gt_x2, gt_y2 = [target_boxes[list(target_boxes.keys())[0]]['x_1'], 
                                          target_boxes[list(target_boxes.keys())[0]]['y_1'], 
                                          target_boxes[list(target_boxes.keys())[0]]['x_2'], 
                                          target_boxes[list(target_boxes.keys())[0]]['y_2']]
            
            # Scale coordinates
            scale_factor = 1000
            gt_x1, gt_y1, gt_x2, gt_y2 = [c * scale_factor for c in [gt_x1, gt_y1, gt_x2, gt_y2]]
            pred_x1, pred_y1, pred_x2, pred_y2 = [c * scale_factor for c in [pred_x1, pred_y1, pred_x2, pred_y2]]
            
            # Create rectangles
            pred_rect = patches.Rectangle((pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1, 
                                        linewidth=3, edgecolor='red', facecolor='none')
            gt_rect = patches.Rectangle((gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1, 
                                      linewidth=3, edgecolor='lime', facecolor='none')
            
            # Add rectangles to plot
            axes[idx].add_patch(pred_rect)
            axes[idx].add_patch(gt_rect)
            
            # Add labels
            pred_keys = list(pred_boxes.keys())
            if any(key in pred_keys for key in ['hand', 'handle']):
                axes[idx].text(pred_x1, pred_y1 - 0.02, 'Pred: hand', color='red', fontsize=7, backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04, 'GT: hand', color='lime', fontsize=7, backgroundcolor='white')
            elif 'index' in pred_keys:
                axes[idx].text(pred_x1, pred_y1 - 0.04, 'Pred: index', color='red', fontsize=7, backgroundcolor='white')
                axes[idx].text(gt_x1, gt_y1 - 0.04, 'GT: index', color='lime', fontsize=7, backgroundcolor='white')
        
        # Add image info and IoU
        axes[idx].text(0.5, -0.08, f"{model_name} - {os.path.basename(row['img_path'])}", 
                      transform=axes[idx].transAxes, ha='center', va='top', fontsize=10)
        axes[idx].text(0.5, -0.11, f"IoU: {row['ious']}", 
                      transform=axes[idx].transAxes, ha='center', va='top', fontsize=10)
    
    # Hide unused subplots
    for j in range(len(plot_data), len(axes)):
        axes[j].axis('off')
    
    # Add title and legend
    plt.suptitle('Good Grasp Prediction Examples', fontsize=45, y=0.99)
    plt.tight_layout()
    
    red_patch = patches.Patch(color='red', label='Predictions')
    green_patch = patches.Patch(color='lime', label='Ground Truth')
    fig.legend(handles=[red_patch, green_patch], loc='upper right', fontsize=25)
    
    plt.show()
    plt.savefig('results/good_prediction_comparison_grid.png')

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
            if 'hand' in ast.literal_eval(row['pred_bboxes']).keys() or 'handle' in ast.literal_eval(row['pred_bboxes']).keys():
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



def plot_bar_chart_by_annotation_average_score(path_list):
    result_dict = {'index_finger': [], 'one_hand': [], 'two_hands': [], 'index_thumb': [], 'handle': []}
    counter = 0
    for path in path_list:
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                iou_dict = ast.literal_eval(row['ious'])
                for key, value in iou_dict.items():
                    if key == 'index' and len(list(iou_dict.keys())) == 1:
                        result_dict['index_finger'].append(value)
                    elif any(substring in row['img_path'] for substring in ['wrench', 'solder', 'screwdriver', 'hammer', 'allen']):
                        result_dict['one_hand'].append(value)
                    elif any(substring in row['img_path'] for substring in ['top', 'bot', 'bar', 'handle', 'door']):
                        result_dict['handle'].append(value)
                        # result_dict['handle'].append(value)
                    elif 'hand1' in list(iou_dict.keys()):
                        result_dict['two_hands'].append(value)
                    elif 'index' in list(iou_dict.keys()) and len(list(iou_dict.keys())) == 2:
                        result_dict['index_thumb'].append(value)
                    else:
                        assert 0 > 1, print(iou_dict, print(iou_dict.keys()), print(row['pred_bboxes']), print(row['img_path']))
                    counter += 1
    print(result_dict['handle'])
    print(f'Number of annotations: {counter}')
    new_result_dict = {'index_finger': sum(result_dict['index_finger'])/len(result_dict['index_finger']), 'one_hand': sum(result_dict['one_hand'])/len(result_dict['one_hand']),
                       'two_hands': sum(result_dict['two_hands'])/len(result_dict['two_hands']), 'index_thumb': sum(result_dict['index_thumb'])/len(result_dict['index_thumb']),
                       'handle': sum(result_dict['handle'])/len(result_dict['handle'])}
    print(len(result_dict['index_finger']), len(result_dict['one_hand']), len(result_dict['two_hands']), len(result_dict['index_thumb']), len(result_dict['handle']))
    print(new_result_dict)
    df = pd.DataFrame.from_dict(new_result_dict, orient='index', columns=['Average IoU'])
    df.index.name = 'Annotation Type in Dataset'
    df = df.reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Annotation Type in Dataset', y='Average IoU', palette='Set1')
    plt.title('Average Intersection over Union by Annotation Type in Dataset', fontsize=16)
    plt.xlabel('Annotation Type in Dataset', fontsize=14)
    plt.ylabel('Average IoU', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.savefig('results/average_iou_by_annotation_type.png', dpi=300, bbox_inches='tight')
                
good_example_list = [('gemini-2.5-flash-lite-preview-06-17', 19), ('gemini-2.5-flash-lite-preview-06-17', 491), ('gemini-2.5-flash-lite-preview-06-17', 461), ('yolo_world', 214), ('yolo_world', 351), ('gemini-2.5-flash-lite-preview-06-17', 371), ('gemini-2.5-flash-lite-preview-06-17', 430), ('yolo_uniow', 215), ('gemini-2.5-flash-lite-preview-06-17', 8)]
bad_example_list = [('grok-2-vision-1212', 263), ('gpt-4.1-nano', 199), ('gpt-4.1-mini', 6), ('owl_vit', 48), ('yolo_uniow', 8), ('yolo_uniow', 167), ('claude-3-haiku-20240307', 263), ('o4-mini', 78), ('o4-mini', 99)]
def plot_ious_vs_tokens(model_list):
    num_models = len(model_list)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axs = axs.flatten()

    for i, path in enumerate(model_list):
        df = pd.read_csv(path, sep=';', encoding='utf-8')
        df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
        model_name = Path(path).stem

        ious = []
        tokens = []
        for _, row in df.iterrows():
            iou_dict = ast.literal_eval(row['ious'])
            for _, value in iou_dict.items():
                ious.append(float(value))
                tokens.append(int(row['output_tokens']))

        ax = axs[i]
        ax.scatter(tokens, ious, alpha=0.6, label='Data Points')

        # Line of best fit
        m, b = np.polyfit(tokens, ious, 1)
        x_vals = np.array(tokens)
        ax.plot(x_vals, m * x_vals + b, color='red', label='Best Fit Line')

        ax.set_title(model_name)
        ax.set_xlabel("Output Tokens")
        ax.set_ylabel("IoU")
        ax.legend()
        ax.grid(True)

    # Hide unused subplots if any
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()
    plt.savefig('results/ious_vs_token_usage_by_model.png', dpi=300, bbox_inches='tight')
    
    # for path in model_list:
    #     df = pd.read_csv(path, sep=';', encoding='utf-8')
    #     df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
    #     for index, row in df.iterrows():
    #         for key, value in ast.literal_eval(row['ious']).items():
    #             ious_to_token_usage_dict.append((value, row['output_tokens']))

    # print(len(ious_to_token_usage_dict))
    # ious, tokens = zip(*ious_to_token_usage_dict)
    # ious = np.array([float(i) for i in ious])
    # tokens = np.array([int(i) for i in tokens])
    # plt.scatter(ious, tokens, alpha=0.6, edgecolor=None, linewidths=0.1)
    # m, b = np.polyfit(ious, tokens, 1)
    # plt.plot(ious, m * ious + b, color='red', linewidth=2, label='Trend Line')
    # plt.xlabel('IoU', fontsize=14)
    # plt.ylabel('Output Tokens', fontsize=14)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('results/ious_vs_token_usage.png', dpi=300, bbox_inches='tight')
    
    
if __name__ == "__main__":
    model_list_selective = ['results/gpt-4.1-mini.csv', 'results/gpt-4.1-nano.csv']
    model_list = ['results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307.csv', 'results/gemini-2.5-flash-lite-preview-06-17.csv',
                                      'results/gemini-2.5-flash.csv', 'results/gemini-2.0-flash-lite.csv', 'results/gpt-4.1-mini.csv', 'results/gpt-4.1-nano.csv',
                                      'results/grok-2-vision-1212.csv', 'results/o4-mini.csv']
    
    # plot_predition_grid(good_example_list, gt_file='ground_truth_test.csv')
    plot_ious_vs_tokens(model_list)

