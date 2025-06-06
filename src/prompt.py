def generate_prompt(tool, annotation_type):
    part_a = f"Given the image of a {tool}, "
    if annotation_type == "index_finger":
        part_b = f"identify the best placement of the pad of the index finger for grasping the {tool} if you had a robotic hand. Focus only on the front surface of the fingertip, not the entire index finger. "
        # part_c = f"Reason through your answer step by step and return your final answer in json as a bounding box around the best placement of the pad of the index finger in the format: [x_min, y_min, x_max, y_max]. After reasoning step by step, return the bounding box as your answer in json."
        part_c = f"Return your final answer in json as a bounding box around the best placement of the pad of the index finger in the format: [x_min, y_min, x_max, y_max]. Return only the bounding box in json as your answer."
    else:
        part_b = f"identify the best region where four fingers would wrap around the {tool} to grasp it if you had a robotic hand. This should be one continous area covering all four fingers. "
        # part_c = f"Reason through your answer step by step and return your final answer in json as a single bounding box around the best placement of the four fingers' contact area in the format: [x_min, y_min, x_max, y_max]. After reasoning step by step, return the bounding box as your answer in json."
        part_c = f"Return your final answer in json as a single bounding box around the best placement of the four fingers' contact area in the format: [x_min, y_min, x_max, y_max]. Return only the bounding box in json as your answer."
    return part_a + part_b + part_c