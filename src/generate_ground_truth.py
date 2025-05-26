#file to generate ground truth csv for vlm testing
import json
import os
import regex as re
import pandas as pd
def generate_ground_truth(json_dir, output_dir):
    full_tool = {'drill' : 'drill', 'wacker' : 'weed_wacker', 'glue' : 'glue_gun', 'saw' : 'circular_saw', 'nail' : 'nail_gun', 
    'screwdriver' : 'screwdriver', 'wrench' : 'wrench', 'solder' : 'solder_iron', 'allen' : 'allen_key', 'hammer' : 'hammer'}
    #some of the tools are shortened in their file names, so these are the full names of the tools
    index_finger = ['circular_saw', 'drill', 'glue_gun', 'nail_gun', 'weed_wacker']
    #these are the tools that correspond to an index finger location bbox
    four_finger = ['allen_key', 'solder_iron', 'wrench', 'screwdriver', 'hammer']
    #these are the tools that correspond to a 4 finger location bbox
    ground_truth_csv_path = os.path.join('src', output_dir)
    #create the csv path in the src directory
    new_df = pd.DataFrame(columns=["img_id", "img_path", "image_dim", "tool", "bbox", "bbox_area", "annotation_type"])
    with open(json_dir, 'r') as f:
        data = json.load(f)
        for i in range(0, 200):
            #we know our dataset has only has 200 images
            img = data['images'][i]
            #json has image ids as a key so we just access that for each image
            img_id = img['id']
            img_path = os.path.join('data/roboflow', img['file_name'])
            img_dim = [img['width'], img['height']]
            #get image dimensions for later observations
            bbox_info = data['annotations'][i]
            #bbox info
            tool_key = re.search(r"drill|wacker|glue|saw|nail|screwdriver|wrench|solder|allen|hammer", img_path).group(0)
            #this is the tool that is in the object
            tool = full_tool[tool_key]
            #retrieve the full tool name for clarity
            bbox_info = data['annotations'][i]
            #bbox annotations are by image id, so we can just index by i
            bbox_x1 = bbox_info['bbox'][0]
            bbox_y1 = bbox_info['bbox'][1]
            bbox_x2 = bbox_x1 + bbox_info['bbox'][2]
            bbox_y2 = bbox_y1 + bbox_info['bbox'][3]
            bbox_dim = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
            # bbox_dim = bbox_info['bbox'][0:4]
            #[x, y, width, height] is coco json(the annotation format), but vlms probably perform better with [x1, y1, x2, y2] so I just convert it
            bbox_area = bbox_info['area']
            #area of bound box for later observations
            annotation_type = "index_finger" if tool in index_finger else "four_finger"
            #index or 4 finger annotation
            print(img_id, img_path, img_dim, tool, bbox_dim, bbox_area, annotation_type)
            new_df.loc[i] = [img_id, img_path, img_dim, tool, bbox_dim, bbox_area, annotation_type]
            new_df.to_csv(ground_truth_csv_path, index=False)
    return ground_truth_csv_path
if __name__ == "__main__":
    generate_ground_truth('data/roboflow/_annotations.coco.json', "ground_truth.csv")
    