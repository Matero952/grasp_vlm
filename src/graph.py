import seaborn as sns
import matplotlib.pyplot as plt
import regex as re
import pandas as pd
import csv
# def box_and_whisker_plot(csv_names):

def get_ious(path_list):
    models_regex = r"claude-3-5-haiku-latest|claude-3-haiku-20240307|gemini-1.5-flash|gemini-2.0-flash|gemini-2.0-flash-lite|gemini-2.5-flash-preview-05-20|grok-2-vision-1212"
    data_dict = {}
    for path in path_list:
        model = re.search(models_regex, path)
        if model:
            if 'reasoning' in model.group(0):
                model_name = f"{model.group(0)}_reasoning"
            else:
                model_name = model.group(0)
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                #decided to use reader and not pd.df because reader is a little bit more lenient on formatting stuff
                ious = []
                counter = 0
                for idx, row in enumerate(reader):
                    counter = 0
                    print(f'{idx}: {row}')
                    print(idx)
                breakpoint()
        #         breakpoint()
        #             if idx != 0:
        #                 if 'No predicted' in str(row[-1]):
        #                     iou = 0
        #                 else:
        #                     try:
        #                         iou = float(row[-1].strip().strip('"'))
        #                     except ValueError:
        #                         iou = 0
        #                     #csv formatting is not the best bc vlm responses are weird so like
        #                 ious.append(iou)
        #             else:
        #                 pass
        #         print(len(ious))
        #         if len(ious) > 200:
        #             print(data_dict)
        #             print(reader.dialect)
        #             breakpoint()
        #             print([row[-1] for row in reader])
        #             print(len([row[-1] for row in reader]))
        #             breakpoint()
        #         data_dict[model_name] = ious
        #     print(data_dict)

        # else:
        #     print("UH OH BREAKING")
        #     breakpoint()
get_ious(['results/claude-3-haiku-20240307/claude-3-haiku-20240307_results_w_reasoning_reformed_maybe.csv', 'results/claude-3-haiku-20240307/claude-3-haiku-20240307_results.csv', 'results/gemini-1.5-flash/gemini-1.5-flash_results_reformed.csv'])