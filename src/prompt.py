
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
    print(f'{annots=}')
    print(coord_format)
    match = re.search(r'-?\d+\.?\d*', file_path)
    if not match:
        print(f"No number found.")
        assert 0 > 1
    file_numb = int(match.group(0))
    #in my files, if the numb was above 10, the green star was the left hand, otherwise it was the right
    print(file_path)
    print(file_numb)
    # breakpoint()
    # print(f'{task=}')
    # breakpoint()
    first_part = f"""
            You are an {vlm_role}. Your objective is to {task}. To do so, you must thoughtfully analyze the provided image of the {tool}
            and identify """
    
    if len(annots) == 1 and annots[0] == 'handle':
        #whole hand grasping
        # print(1)
        second_part = f"""the best region where your robotic hand would wrap around the {tool} to use it. \nAssume that the {tool} can be immediately used and that your hand has one continuous contact region with the {tool}."""
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"hand" : bounding_box}}, \nwhere the bounding box follows this format: \n{coord_format} ' 
    elif len(annots) == 1 and annots[0] == 'index':
        #index finger grasping
        # print(2)
        second_part = f"""the best placement of the pad of your robotic index finger for grasping and using the {tool}. \nAssume that the {tool} can be immediately used and that your fingertip has one continuous contact region with the {tool}. \nFocus only on the front surface of the fingertip, not the entire index finger."""
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"index" : bounding_box}}, \nwhere the bounding box follows this format: \n{coord_format} ' 
        pass
    elif len(annots) == 2 and 'left' in annots:
        #left hand right hand grasping
        assert 'right' in annots
        second_part = f"""the best placements of your left and right robotic hands to grasp and use the {tool}."""
        if file_numb > 10:
            second_part += ' A green star is placed in the image near the area that should be grasped by the LEFT hand. It serves as a visual reference point, not the exact contact region. \n'
        elif file_numb < 11:
            second_part += ' A green star is placed in the image near the area that should be grasped by the RIGHT hand. It serves as a visual reference point, not the exact contact region. \n'
        else:
            assert 0 > 1
        second_part += f'Assume that the {tool} can be immediately used and each hand has its own continuous contact region with the {tool}.'
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"left" : bounding_box_a, right : bounding_box_b}}, \nwhere the bounding boxes follow this format: \n{coord_format} '  
        # print(3)
    elif len(annots) == 2 and 'index' in annots:
        #index thumb grasping
        assert 'thumb' in annots
        second_part = f"""the best placements of your robotic index finger and thumb on the {tool}. \nAssume that the {tool} can be immediately and each finger has its own continuous contact region with the {tool}. \nFocus on the contact region between the finger and the {tool}. """
        third_part = f' Your output needs to contain a JSON object structured like this: \n {{"index" : bounding_box_a, "thumb": bounding_box_b}}, \nwhere the bounding boxes follow this format: \n{coord_format} ' 
        # print(4)
    fourth_part = 'Be accurate and think/reason your answer step by step.'
    return first_part + second_part + third_part + fourth_part
    
    
# def generate_prompt(tool, annotation_type):
#     part_a = f"Given the image of a {tool}, "
#     if annotation_type == "index_finger":
#         part_b = f"identify the best placement of the pad of the index finger for grasping the {tool} if you had a robotic hand. Focus only on the front surface of the fingertip, not the entire index finger. "
#         # part_c = f"Reason through your answer step by step and return your final answer in json as a bounding box around the best placement of the pad of the index finger in the format: [y_min, x_min, y_max, x_max]. After reasoning step by step, return the bounding box as your answer in json."
#         part_c = f"Return your final answer in json as a bounding box around the best placement of the pad of the index finger in the format: [x_min, y_min, x_max, y_max]. Return only the bounding box in json as your answer."
#         # part_c_grok = f"Reason through your answer step by step and return your final answer in json as a bounding box around the best placement of the pad of the index finger. The coordinates of the bound box should be normalized relative to the width and height of the object, where the top left corner is (0,0), the bottom right corner is (1,1), x coordinates increase from left to right, y coordinates increase from top to bottom, and the bound box should be in the format: [top_left_x, top_left_y, bot_right_x, bot_right_y]. After reasoning step by step, return the bounding box as your answer in json."

#     else:
#         part_b = f"identify the best region where four fingers would wrap around the {tool} to grasp it if you had a robotic hand. This should be one continous area covering all four fingers. "
#         # part_c = f"Reason through your answer step by step and return your final answer in json as a single bounding box around the best placement of the four fingers' contact area in the format: [y_min, x_min, y_max, x_max]. After reasoning step by step, return the bounding box as your answer in json."
#         part_c = f"Return your final answer in json as a single bounding box around the best placement of the four fingers' contact area in the format: [x_min, y_min, x_max, y_max]. Return only the bounding box in json as your answer."
#         # part_c_grok = f"Reason through your answer step by step and return your final answer in json as a bounding box around the best placement of the four fingers. The coordinates of the bound box should be normalized relative to the width and height of the object, where the top left corner is (0,0), the bottom right corner is (1,1), x coordinates increase from left to right, y coordinates increase from top to bottom, and the bound box should be in the format: [top_left_x, top_left_y, bot_right_x, bot_right_y]. After reasoning step by step, return the bounding box as your answer in json."
#         #grok cookbook suggests to have bound boxes outputted like this
#     return part_a + part_b + part_c

if __name__ == '__main__':
    df = pd.read_csv('ground_truth_test.csv', sep=';')
    for i in range(100):
        row = df.iloc[i]
        bbox = ast.literal_eval(row['bboxes'])
        file_path = str(row['img_path'])
        tool = str(row['tool'])
        task = str(row['task'])
        role = str(row['vlm_role'])
        prompt = get_prompt(file_path, tool, role, 'gemini-2.0-flash', task, bbox)
        print(f'{prompt}')
        print(f'{row}')
        breakpoint()
