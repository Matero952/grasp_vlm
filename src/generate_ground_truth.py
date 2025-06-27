#file to generate ground truth csv for vlm testing
import json
import os
import regex as re
import pandas as pd
tool_dict = {'bot_bar' : ('bottom drawer with a horizontal bar handle', 'AI household robot', 'put away folded clothing into the bottom drawer with a horizontal bar handle'),
             'top_bar' : ('top drawer with a horizontal bar handle', 'AI household robot', 'put away folded clothing into the top drawer with a horizontal bar handle'),
             'lever_handle' : ('door with a horizontal lever handle', 'AI household robot', 'reach the horizontal lever doorhandle'),
             'rnd_door' : ('door with a round handle', 'AI household robot', 'reach the round doorhandle'),
             'bot_rnd' : ('bottom drawer with a round handle', 'AI household robot', 'put away folded clothing into the bottom drawer with a round handle'),
             'top_rnd' : ('top drawer with a round handle', 'AI household robot', 'put away folded clothing into the top drawer with a round handle'),
             'vertical_bar' : ('door with a vertical bar handle', 'AI household robot', 'reach the vertical bar doorhandle'),
             'bowl_ball' : ('bowling ball', 'AI bowling robot', 'bowl a strike'),
             'darts' : ('dart', 'AI dart-throwing robot', 'throw a bullseye'),
             'scissors' : ('pair of scissors', 'AI arts-and-crafts robot', 'cut some paper'),
             'syringe' : ('syringe', 'AI medical robot', 'vaccinate a patient'),
             'bow' : ('violin bow', 'AI virtuoso violinist robot', 'hold your bow'),
             'bolt_cutters' : ('bolt cutters', 'AI construction robot', 'cut a padlock'),
             'can_opener' : ('can opener', 'AI cooking robot', 'open a can'),
             'chainsaw' : ('chainsaw', 'AI landscaping robot', 'cut firewood'),
             'multimeter' : ('multimeter with red and black leads', 'AI electrician robot', 'test continuity on a low-voltage circuit'),
             'shovel' : ('shovel', 'AI landscaping robot', 'dig a hole'),
             'allen' : ('allen key', 'AI household robot', 'tighten the bolts on a chair'),
             'saw' : ('circular saw', 'AI construction robot', 'safely cut a wooden plank'),
             'drill' : ('power drill', 'AI construction robot', 'safely drill a hole in drywall'),
             'glue' : ('glue gun', 'AI arts-and-crafts robot', 'glue two pieces of cardboard together'),
             'hammer' : ('hammer', 'AI construction robot', 'safely hammer a nail into wood'),
             'nail' : ('nail gun', 'AI construction robot', 'safely use the nailgun on a wood surface'),
             'screwdriver' : ('screwdriver', 'AI household robot', 'fasten a screw on a chair'),
             'solder' : ('soldering iron', 'AI electronic technician robot', 'safely make a solder connection'),
             'wacker' : ('weed wacker', 'AI landscaping robot', 'trim the grass'),
             'wrench' : ('wrench', 'AI household robot', 'loosen a nut on a chair')
             }
def generate_models_card(output_path):
    models = ['claude-3-5-haiku-latest', 'claude-3-haiku-20240307',
              'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-flash-preview-tts', 'gemini-2.0-flash-lite',
              'grok-2-vision-1212', 'o4-mini', 'gpt-4o-mini', 'gpt-4.1-nano']
    models_card = {}
    models_card['claude-3-5-haiku-latest'] = 'Normalized coordinates (0-1): [x1, y1, x2, y2]. First point (x1,y1) is top-left, second point (x2,y2) is bottom-right. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['claude-3-haiku-20240307'] = 'Normalized coordinates (0-1): [x1, y1, x2, y2]. First point (x1,y1) is top-left, second point (x2,y2) is bottom-right. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['o4-mini'] = 'Normalized coordinates (0-1): [x1, y1, x2, y2]. First point (x1,y1) is top-left, second point (x2,y2) is bottom-right. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['gpt-4.1-mini'] = 'Normalized coordinates (0-1): [x1, y1, x2, y2]. First point (x1,y1) is top-left, second point (x2,y2) is bottom-right. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['gpt-4.1-nano'] = 'Normalized coordinates (0-1): [x1, y1, x2, y2]. First point (x1,y1) is top-left, second point (x2,y2) is bottom-right. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['grok-2-vision-1212'] = 'Normalized coordinates (0-1): [x1, y1, x2, y2]. First point (x1,y1) is top-left, second point (x2,y2) is bottom-right. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['gemini-1.5-flash'] = 'Normalized coordinates (0-1000): [y_min, x_min, y_max, x_max]. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['gemini-2.5-flash-preview-tts'] = 'Normalized coordinates (0-1000): [y_min, x_min, y_max, x_max]. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['gemini-2.0-flash-lite'] = 'Normalized coordinates (0-1000): [y_min, x_min, y_max, x_max]. Origin (0,0) at top-left, y-axis increases downward.'
    models_card['gemini-2.0-flash'] = 'Normalized coordinates (0-1000): [y_min, x_min, y_max, x_max]. Origin (0,0) at top-left, y-axis increases downward.'    
    models_card['gemini-2.5-flash-lite-preview-06-17'] = 'Normalized coordinates (0-1000): [y_min, x_min, y_max, x_max]. Origin (0,0) at top-left, y-axis increases downward.' 
    models_card['gemini-2.5-pro'] = 'Normalized coordinates (0-1000): [y_min, x_min, y_max, x_max]. Origin (0,0) at top-left, y-axis increases downward.' 
    models_card['gemini-2.5-flash'] = 'Normalized coordinates (0-1000): [y_min, x_min, y_max, x_max]. Origin (0,0) at top-left, y-axis increases downward.'
    df = pd.DataFrame.from_dict(models_card, orient='index', columns=['coordinate_format'])
    # Reset index to make model names a regular column
    df = df.reset_index().rename(columns={'index': 'model'})
    df.to_csv(output_path, sep=';', encoding='utf-8', index=False)
def generate_ground_truth(rf_json_path, output_path):
    category_mapping = {}
    rows = []
    #{img_id: , img_path: , img_width: , img_height: , tool: , vlm_role: , task, bboxes: {bbox1: , bbox2: }}
    with open(rf_json_path, 'r') as f:
        dataset = json.load(f)
        for category_id in dataset['categories']:
            if category_id['id'] != 0:
                #Two category ids, '0' and '1' map to the same thing in the json
                category_mapping[category_id['id']] = category_id['name']
    print(category_mapping)
    for img_data in dataset['images']:
        row = {}
        row['img_id'] = img_data['id']
        row['img_path'] = os.path.join('grasp_vlm_dataset', img_data['file_name'])
        row['img_width'] = img_data['width']
        row['img_height'] = img_data['height']
        tool_key = re.split(r'(?=[0-9])|(?=_\d+_)|(?=_jpg)', img_data['file_name'], maxsplit=1)[0]
        #regex splits file name like: 'syringe18_jpg.rf.e2aabf9a930932ecfa1a9fd7355ebde7.jpg' into
        #['syringe', '18_jpg.rf.e2aabf9a930932ecfa1a9fd7355ebde7.jpg']
        print(tool_key)
        tool, vlm_role, task = tool_dict[tool_key]
        row['tool'] = tool
        row['vlm_role'] = vlm_role
        row['task'] = task
        print(tool)
        annotations = [annot for annot in dataset['annotations'] if annot.get('image_id') == img_data['id']]
        print(annotations)
        bboxes = {}
        for bbox in annotations:
            grasp_type = category_mapping[bbox['category_id']]
            x_min = bbox['bbox'][0]
            y_min = bbox['bbox'][1]
            x_max = x_min + bbox['bbox'][2]
            y_max = y_min + bbox['bbox'][3]
            #normalize al coordinates 
            x_min = x_min / int(img_data['width'])
            x_max = x_max / int(img_data['width'])
            y_min = y_min / int(img_data['height'])
            y_max = y_max / int(img_data['height'])
            #save in standard format: top-left (x_1, y_1) to bottom-right (x_2, y_2)
            bboxes[grasp_type] = {'x_1': x_min,'y_1': y_min,'x_2': x_max,'y_2': y_max}
        row['bboxes'] = dict(sorted(bboxes.items()))
        print(row)
        rows.append(row)
    print(rows)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep=';', encoding='utf-8', index=False)
if __name__ == "__main__":
    generate_ground_truth('grasp_vlm_dataset/_annotations.coco.json', "ground_truth_test.csv")
    generate_models_card('model_cards.csv')
    