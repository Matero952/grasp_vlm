import torch
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import numpy as np
from torchvision.ops import clip_boxes_to_image, remove_small_boxes
import json
import os

import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
    
_script_dir = os.path.dirname(os.path.realpath(__file__))
_config_path = os.path.join(_script_dir, 'config.json')
fig_dir = os.path.join(_script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(os.path.join(fig_dir, "SAM2"), exist_ok=True)
os.makedirs(os.path.join(fig_dir, "OWLV2"), exist_ok=True)
config = json.load(open(_config_path, 'r'))

class OWLv2:
    def __init__(self):
        """
        Initializes the OWLv2 model and processor.
        Parameters:
        - iou_th: IoU threshold for NMS
        - discard_percentile: percentile to discard low scores
        """
        # Load the OWLv2 model and processor
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device

        # Move model to the appropriate device
        self.model.to(self.device)
        self.model.eval()  # set model to evaluation mode


    def predict(self, img, querries, debug = False):
        """
        Gets realsense frames
        Parameters:
        - img: image to produce bounding boxes in
        - querries: list of strings whos bounding boxes we want
        - debug: if True, prints debug information
        Returns:
        - out_dict: dictionary containing a list of bounding boxes and a list of scores for each query
        """
        #Preprocess inputs
        print(f"{img.shape=}")
        inputs = self.processor(text=querries, images=img, return_tensors="pt", do_rescale=False)
        inputs.to(self.device)

        #model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        # print(f"img shape in woL: {img.shape}")
        # print(img.shape[:2])
        # target_sizes = torch.tensor([img.shape[:2]])  # (height, width)
        channels, height, width = img.shape
        target_sizes = torch.tensor([[height, width]], dtype=torch.float32)
        target_sizes_2 = torch.tensor([img.shape[:2]])
        # target_sizes = torch.tensor([img.shape[:2]], dtype=torch.float32)

        print(f"target_sizes 1: {target_sizes}")
        print(f"target sizes 2: {target_sizes_2}")
        #breakpoint()


        results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0)[0]
        label_lookup = {i: label for i, label in enumerate(querries)}
        all_labels = results["labels"]
        all_boxes = results["boxes"]
        print(f"{all_boxes=}")
        all_scores = results["scores"]

        all_boxes = clip_boxes_to_image(all_boxes, (height, width))
        keep = remove_small_boxes(all_boxes, min_size=config["min_2d_box_side"])
        small_removed_boxes = all_boxes[keep]
        small_removed_scores = all_scores[keep]
        small_removed_labels = all_labels[keep]

        
        out_dict = {}
        #for each query, get the boxes and scores and perform NMS
        for i, label in enumerate(querries):
            text_label = label_lookup[i]

            # Filter boxes and scores for the current label
            mask = small_removed_labels == i
            instance_boxes = small_removed_boxes[mask]
            instance_scores = small_removed_scores[mask]
            out_dict[text_label] = {"scores": instance_scores, "boxes": instance_boxes}
        for k, v in out_dict.items():
            print(f"{k=}, {v=}")
        return out_dict

if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2

    # print('Starting OWL test')
    # owl = OWLv2()
    # img = Image.open('grasp_vlm_dataset/shovel1_jpg.rf.3a96b405cb30873a37faf65b8e32223d.jpg')
    # img = T.ToTensor()(img)
    # prompt = ['shovel']
    # result_dict = owl.predict(img, prompt)
    # print(f'{result_dict=}')
    # best_box = result_dict[prompt[0]]['boxes'][torch.argmax(result_dict[prompt[0]]['scores'])]
    # best_box = best_box.tolist()
    # print(f'{best_box=}')
    # # Load image (using cv2 or any other method)
    best_box = [144, 526, 975, 622]
    # best_box = [144, 526, 831, 96]
    img = cv2.imread('grasp_vlm_dataset/allen_1_jpg.rf.a73ee009d63cee3f706aae5f396f4180.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

    # Define bounding box coordinates
    # Format: [xmin, ymin, xmax, ymax]

    # Create plot
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Create a Rectangle patch
    rect = patches.Rectangle(
        (best_box[0], best_box[1]),            # (x, y)
        best_box[2] - best_box[0],             # width
        best_box[3] - best_box[1],             # height
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    # Add the rectangle to the Axes
    ax.add_patch(rect)

    plt.axis('off')
    plt.show()
