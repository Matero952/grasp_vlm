
def generate_prompt(tool, annotation_type):
    part_a = f"Given the image of a {tool}, "
    if annotation_type == "index_finger":
        part_b = f"identify the best placement of the pad of the index finger for grasping the {tool} if you had a robotic hand. Focus only on the front surface of the fingertip, not the entire index finger. "
        # part_c = f"Reason through your answer step by step and return your final answer in json as a bounding box around the best placement of the pad of the index finger in the format: [y_min, x_min, y_max, x_max]. After reasoning step by step, return the bounding box as your answer in json."
        part_c = f"Return your final answer in json as a bounding box around the best placement of the pad of the index finger in the format: [x_min, y_min, x_max, y_max]. Return only the bounding box in json as your answer."
        # part_c_grok = f"Reason through your answer step by step and return your final answer in json as a bounding box around the best placement of the pad of the index finger. The coordinates of the bound box should be normalized relative to the width and height of the object, where the top left corner is (0,0), the bottom right corner is (1,1), x coordinates increase from left to right, y coordinates increase from top to bottom, and the bound box should be in the format: [top_left_x, top_left_y, bot_right_x, bot_right_y]. After reasoning step by step, return the bounding box as your answer in json."

    else:
        part_b = f"identify the best region where four fingers would wrap around the {tool} to grasp it if you had a robotic hand. This should be one continous area covering all four fingers. "
        # part_c = f"Reason through your answer step by step and return your final answer in json as a single bounding box around the best placement of the four fingers' contact area in the format: [y_min, x_min, y_max, x_max]. After reasoning step by step, return the bounding box as your answer in json."
        part_c = f"Return your final answer in json as a single bounding box around the best placement of the four fingers' contact area in the format: [x_min, y_min, x_max, y_max]. Return only the bounding box in json as your answer."
        # part_c_grok = f"Reason through your answer step by step and return your final answer in json as a bounding box around the best placement of the four fingers. The coordinates of the bound box should be normalized relative to the width and height of the object, where the top left corner is (0,0), the bottom right corner is (1,1), x coordinates increase from left to right, y coordinates increase from top to bottom, and the bound box should be in the format: [top_left_x, top_left_y, bot_right_x, bot_right_y]. After reasoning step by step, return the bounding box as your answer in json."
        #grok cookbook suggests to have bound boxes outputted like this
    return part_a + part_b + part_c