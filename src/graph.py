#graphing functions, redo box and whiskers, add graphs by image size maybe?
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import regex as re
import math
from itertools import cycle
matplotlib.use('AGG')
tool_dict = {'drill' : 'drill', 'wacker' : 'weed_wacker', 'glue' : 'glue_gun', 'saw' : 'circular_saw', 'nail' : 'nail_gun', 
    'screwdriver' : 'screwdriver', 'wrench' : 'wrench', 'solder' : 'solder_iron', 'allen' : 'allen_key', 'hammer' : 'hammer'}
models_regex = r"claude-3-5-haiku-latest|claude-3-haiku-20240307|gemini-1.5-flash|gemini-2.0-flash|gemini-2.0-flash-lite|gemini-2.5-flash-preview-05-20|grok-2-vision-1212"

def plot_box_and_whiskers(iou_dict: dict):
    # df = pd.DataFrame({key: value for key, value in iou_dict.items()
    # })
    df = pd.DataFrame(iou_dict)
    df = df.melt(var_name="Model", value_name="IoU")
    plt.figure(figsize=(16, 15))
    sns.boxplot(x = 'IoU', y = 'Model', data=df)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Iou', fontsize=20)
    plt.ylabel('Model', fontsize=20)
    plt.title('Boxplot of IoUs for All Models', fontsize=25)
    plt.tight_layout()
    plt.show()
    plt.savefig('src/iou_boxplt.png')

def plot_prediction_grid(csv_path, numb_of_imgs, gt_file='src/ground_truth.csv'):
    df = pd.read_csv(csv_path, sep=';')
    gt = pd.read_csv(gt_file, sep=';')
    df.columns = df.columns.str.replace('"', '', regex=False)
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
    for i in range(len(img_paths_by_tool.keys())):
        key = next(key_cycle)
        values = img_paths_by_tool[key][:amount_per_tool]
        for i in values:
            imgs.append(i)

    for j in range(left_over):
        key = next(key_cycle)
        imgs.append(img_paths_by_tool[key][len(img_paths_by_tool.keys()) + 1])

    #check for repeats
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
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    for idx, img_path in enumerate(imgs):
        img = mpimg.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')  # Turn off axes
        to_check_row_gt = gt[gt['img_path'].astype(str) == str(img_path)]
        img_id = to_check_row_gt['img_id']
        to_check_row_df = df[df['img_id'].astype(str) == str(img_id)]
        iou = to_check_row_df['iou']
        axes[idx].set_title(f"ID: {img_id}, IoU: {iou}")

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('src/test.png')


def get_ious(csv_list):
    ious = {}
    for path in csv_list:
        df = pd.read_csv(path, delimiter=';')
        #I decided to change the delimiter to semicolon as none of the vlm resposnes used it and it looks a little bit cleaner in my opinion.
        df.columns = df.columns.str.replace('"', '', regex=False)
        model_match = re.search(models_regex, path)
        if model_match:
            if 'reason' in path:
                model_name = f"{model_match.group(0)}_reasoning"
            else:
                model_name = model_match.group(0)
            ious[model_name] = [float(i) for i in df['iou'].to_list()]
        else:
            breakpoint()
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
    # plot_box_and_whiskers(get_ious(['results/claude-3-5-haiku-latest-reasoning_reason.csv', 'results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307-reasoning_reason.csv', 
    #                 'results/claude-3-haiku-20240307.csv', 'results/gemini-1.5-flash-reasoning_reason.csv', 'results/gemini-1.5-flash.csv', 'results/gemini-2.0-flash-lite-reasoning_reason.csv',
    #                 'results/gemini-2.0-flash-lite.csv', 'results/gemini-2.0-flash-reasoning_reason.csv', 'results/gemini-2.0-flash.csv', 'results/gemini-2.5-flash-preview-05-20-reasoning_reason.csv',
    #                 'results/gemini-2.5-flash-preview-05-20.csv', 'results/grok-2-vision-1212-reasoning_reason.csv', 'results/grok-2-vision-1212.csv'
    #                 ]))
    plot_prediction_grid('results/claude-3-5-haiku-latest-reasoning_reason.csv', 64)
    # get_img_paths_by_tool('src/ground_truth.csv')





