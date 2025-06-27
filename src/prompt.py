
import pandas as pd
import ast
import re

def get_prompt(file_path, tool, vlm_role, model, task, bboxes: dict):
    #tool for the actual tool in the picture, vlm role for the scenario for the vlm, explicit bbox format for the vlm,
    #bboxes for the specific fingers/hands that need to be annotated(index + thumb vs left and right hand)
    df = pd.read_csv('model_cards.csv', sep=';', encoding='utf-8')
    df.columns = df.columns.str.replace('"', '', regex=False)
    to_check_row = df[df['model'] == str(model)]
    coord_format = to_check_row['coordinate_format'].iloc[0]
    annots = list(bboxes.keys())
    print(file_path)
    first_part = f"""
            You are an {vlm_role}. Your objective is to {task}. To do so, you must thoughtfully analyze the provided image of the {tool}
            and identify """
    if len(annots) == 1 and annots[0] == 'handle':
        second_part = f"""the best region where your robotic hand would wrap around the {tool} to use it. \nAssume that the {tool} can be immediately used and that your hand has one continuous contact region with the {tool}."""
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"hand" : bounding_box}}, \nwhere the bounding box follows this format: \n{coord_format} ' 
    elif len(annots) == 1 and annots[0] == 'index':
        #index finger grasping
        # print(2)
        second_part = f"""the best placement of the pad of your robotic index finger for grasping and using the {tool}. \nAssume that the {tool} can be immediately used and that your fingertip has one continuous contact region with the {tool}. \nFocus only on the front surface of the fingertip, not the entire index finger."""
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"index" : bounding_box}}, \nwhere the bounding box follows this format: \n{coord_format} ' 
        pass
    elif len(annots) == 2 and 'hand1' in annots:
        #left hand right hand grasping
        assert 'hand2' in annots
        second_part = f"""the best placements of your left and right robotic hands to grasp and use the {tool}."""
        second_part += f' Assume that the {tool} can be immediately used and each hand has its own continuous contact region with the {tool}.'
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"hand1" : bounding_box_a, "hand2" : bounding_box_b}}, \nwhere the bounding boxes follow this format: \n{coord_format} '  
        # print(3)
    elif len(annots) == 2 and 'index' in annots:
        #index thumb grasping
        assert 'thumb' in annots
        second_part = f"""the best placements of your robotic index finger and thumb on the {tool}. \nAssume that the {tool} can be immediately and each finger has its own continuous contact region with the {tool}. \nFocus on the contact region between the finger and the {tool}. """
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"index" : bounding_box_a, "thumb": bounding_box_b}}, \nwhere the bounding boxes follow this format: \n{coord_format} ' 
        # print(4)
    fourth_part = 'Be accurate and think/reason your answer step by step.'
    # print(first_part)
    # print(second_part)
    return first_part + second_part + third_part + fourth_part
    

if __name__ == '__main__':
    df = pd.read_csv('ground_truth_test.csv', sep=';')
    for i in range(100):
        row = df.iloc[i]
        bbox = ast.literal_eval(row['bboxes'])
        file_path = str(row['img_path'])
        tool = str(row['tool'])
        task = str(row['task'])
        role = str(row['vlm_role'])
        prompt = get_prompt(file_path, tool, role, 'claude-3-5-haiku-latest', task, bbox)
        print(f'{prompt}')
        print(f'{row}')
        breakpoint()
