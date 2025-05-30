#graphing functions, redo box and whiskers, add graphs by image size maybe?
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import regex as re
matplotlib.use('AGG')

models_regex = r"claude-3-5-haiku-latest|claude-3-haiku-20240307|gemini-1.5-flash|gemini-2.0-flash|gemini-2.0-flash-lite|gemini-2.5-flash-preview-05-20|grok-2-vision-1212"
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

plot_box_and_whiskers(get_ious(['results/claude-3-5-haiku-latest-reasoning_reason.csv', 'results/claude-3-5-haiku-latest.csv', 'results/claude-3-haiku-20240307-reasoning_reason.csv', 
                  'results/claude-3-haiku-20240307.csv', 'results/gemini-1.5-flash-reasoning_reason.csv', 'results/gemini-1.5-flash.csv', 'results/gemini-2.0-flash-lite-reasoning_reason.csv',
                  'results/gemini-2.0-flash-lite.csv', 'results/gemini-2.0-flash-reasoning_reason.csv', 'results/gemini-2.0-flash.csv', 'results/gemini-2.5-flash-preview-05-20-reasoning_reason.csv',
                  'results/gemini-2.5-flash-preview-05-20.csv', 'results/grok-2-vision-1212-reasoning_reason.csv', 'results/grok-2-vision-1212.csv'
                  ]))





