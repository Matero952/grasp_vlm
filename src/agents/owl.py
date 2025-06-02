import torch
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import warnings
import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from open3d.visualization import gui, rendering

import numpy as np
from torchvision.ops import clip_boxes_to_image, remove_small_boxes
import json
import random
import os

import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
from utils import get_points_and_colors, nms



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
        inputs = self.processor(text=querries, images=img, return_tensors="pt", do_rescale=False)
        inputs.to(self.device)

        #model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.shape[:2]])  # (height, width)

        results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0)[0]

        # Extract labels, boxes, and scores
        label_lookup = {i: label for i, label in enumerate(querries)}
        all_labels = results["labels"]
        all_boxes = results["boxes"]
        all_boxes = clip_boxes_to_image(all_boxes, img.shape[:2])
        all_scores = results["scores"]
        if debug:
            temp_pred = {}
            for i, label in enumerate(querries):
                mask = all_labels == i
                text_label = label_lookup[i]
                temp_pred[text_label] = {"boxes": all_boxes[mask], "scores": all_scores[mask]}
            display_owl(img, temp_pred, window_prefix=f"All Boxes")

        keep = remove_small_boxes(all_boxes, min_size=config["min_2d_box_side"])
        small_removed_boxes = all_boxes[keep]
        small_removed_scores = all_scores[keep]
        small_removed_labels = all_labels[keep]
        if debug:
            temp_pred = {}
            for i, label in enumerate(querries):
                mask = small_removed_labels == i
                text_label = label_lookup[i]
                temp_pred[text_label] = {"boxes": small_removed_boxes[mask], "scores": small_removed_scores[mask]}
            display_owl(img, temp_pred, window_prefix=f"Small Boxes Removed")

        #get integer to text label mapping
        out_dict = {}
        #for each query, get the boxes and scores and perform NMS
        for i, label in enumerate(querries):
            text_label = label_lookup[i]

            # Filter boxes and scores for the current label
            mask = small_removed_labels == i
            instance_boxes = small_removed_boxes[mask]
            instance_scores = small_removed_scores[mask]

            #Do NMS for the current label
            pruned_boxes, pruned_scores, _ = nms(instance_boxes.cpu(), instance_scores.cpu(), iou_threshold=config["iou_2d_reduction"], three_d=False)
            pruned_boxes  = torch.stack(pruned_boxes)
            pruned_scores = torch.stack(pruned_scores)

            if debug:
                display_owl(img, {text_label: {"boxes": pruned_boxes, "scores": pruned_scores}}, window_prefix=f"Post NMS ")
            #print(f"{pruned_boxes.shape=}, {pruned_scores.shape=}")

            #Get rid of low scores
            threshold = torch.quantile(pruned_scores, config["owlv2_discard_percentile"])
            keep = pruned_scores > threshold
            filtered_boxes  = pruned_boxes[keep]
            filtered_scores = pruned_scores[keep]

            # Normalize scores
            filtered_scores = F.sigmoid(filtered_scores*config["sigmoid_gain"])
            #mu    = filtered_scores.mean()
            #sigma = filtered_scores.std(unbiased=False)
            #z = (filtered_scores - mu) / sigma
            #filtered_scores = F.softmax(z, dim=0)
            # filtered_scores = filtered_scores / filtered_scores.sum()

            # Update output dictionary
            out_dict[text_label] = {"scores": filtered_scores, "boxes": filtered_boxes}

            if debug:
                print(f"{text_label=}")
                print(f"    {all_labels.shape=}, {all_boxes.shape=}, {all_scores.shape=}")
                print(f"    {small_removed_boxes.shape=}, {small_removed_scores.shape=}, {small_removed_labels.shape=}")
                print(f"    {instance_scores.shape=}, {instance_boxes.shape=}")
                print(f"    {pruned_boxes.shape=}, {pruned_scores.shape=}")
                print(f"    {len(keep)=}")
                print(f"    {filtered_scores.shape=}, {filtered_boxes.shape=}")
                print(f"    {threshold=}")
                print()
                display_owl(img, {text_label: {"boxes": filtered_boxes, "scores": filtered_scores}}, window_prefix=f"Top {config['owlv2_discard_percentile']} quantile ")


        if debug:
            for key in out_dict:
                print(f"{key=} {out_dict[key]['boxes'].shape=}, {out_dict[key]['scores'].shape=}")
        return out_dict, inputs['pixel_values'].shape

    def __str__(self):
        return f"OWLv2: {self.model.device}"
    def __repr__(self):
        return self.__str__()
def display_owl(img, predicitons, window_prefix = ""):
    for query_object, prediction in predicitons.items():
        display_img = img.copy()
        for bbox, score in zip(prediction["boxes"], prediction["scores"]):
            #bbox = prediction["box"]
            #score = prediction["score"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, f"{query_object} {score:.4f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the image with bounding boxes
        cv2.imwrite(f"{fig_dir}/OWLV2/{window_prefix}{query_object}.png", display_img)
        cv2.imshow(f"{window_prefix}{query_object}", display_img)
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def test_OWL(left_img, debug):
    owl = OWLv2()
    predicitons = owl.predict(left_img, config["test_querys"], debug=debug)
    display_owl(left_img, predicitons)
    return predicitons
# Class to use sam2
# class SAM2_PC:
#     def __init__(self,):
#         """
#         Initializes the SAM2 model and processor.
#         Parameters:
#         - iou_th: IoU threshold for NMS
#         """
#         self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device
#     def predict(self, rgb_img, depth_img, bbox, scores, intrinsics, debug = False):
#         """
#         Predicts 3D point clouds from RGB and depth images and bounding boxes using SAM2.
#         Cleans up the point clouds and applies NMS.
#         Parameters:
#         - rgb_img: RGB image
#         - depth_img: Depth image
#         - bbox: Bounding boxes
#         - scores: Scores for the bounding boxes
#         - intrinsics: Camera intrinsics
#         - debug: If True, prints debug information
#         Returns:
#         - reduced_pcds: List of reduced point clouds
#         - reduced_bounding_boxes_3d: List of reduced 3D bounding boxes
#         - reduced_scores: List of reduced scores
#         """
#         #Run sam2 on all the boxes
#         self.sam_predictor.set_image(rgb_img.copy())
#         sam_mask = None
#         sam_scores = None
#         sam_logits = None
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UserWarning)
#             sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box=bbox)
#         sam_mask = np.all(sam_mask, axis=1)
#         if debug:
#             print(f"{sam_mask.shape=}")


#         #Apply mask to the depth and rgb images
#         masked_depth = depth_img[None, ...] * sam_mask
#         masked_rgb = rgb_img[None, ...] * sam_mask[..., None]
#         #print(f"\n\n{masked_depth.shape=}, {masked_rgb.shape=}")
#         #print(f"{type(masked_depth)=}, {type(masked_rgb)=}\n\n")
#         if debug:
#             n_fig = 2
#             print(f"{masked_depth.shape=}, {masked_rgb.shape=}")
#             fig, ax = plt.subplots(n_fig, 4, figsize=(20, 10))
#             ax[0,0].set_title("RGB")
#             ax[0,1].set_title("Depth")
#             ax[0,2].set_title("Masked RGB")
#             ax[0,3].set_title("Masked Depth")
#             for i in range(min(n_fig, sam_mask.shape[0])):
#                 bbox_rgb = rgb_img.copy()
#                 bbox_depth = depth_img.copy()
#                 x_min, y_min, x_max, y_max = map(int, bbox[i])
#                 cv2.rectangle(bbox_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 cv2.rectangle(bbox_depth, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 ax[i, 0].imshow(bbox_rgb)
#                 ax[i, 1].imshow(bbox_depth)
#                 ax[i, 2].imshow(masked_rgb[i])
#                 ax[i, 3].imshow(masked_depth[i])
#             fig.tight_layout()
#             fig.savefig(f"{fig_dir}/SAM2/masks_{random.randint(0,100)}.png")
#             plt.show()

#         #Get points and colors from masked depth and rgb images
#         #print("here 1")
#         tensor_depth = torch.from_numpy(masked_depth).to(self.device)
#         tensor_rgb = torch.from_numpy(masked_rgb).to(self.device)
#         #print("here 2")

#         fx = intrinsics["fx"]
#         fy = intrinsics["fy"]
#         cx = intrinsics["cx"]
#         cy = intrinsics["cy"]
#         points, colors = get_points_and_colors(tensor_depth, tensor_rgb, fx, fy, cx, cy)
#         colors = colors[..., [2, 1, 0]]
#         if debug:
#             print(f"{points.shape=}, {colors.shape=}")

#         B, N, _ = points.shape

#         full_pcds = []
#         full_bounding_boxes_3d = []
#         full_scores = []
#         full_masked_rgb = []
#         full_masked_depth = []

#         pts_cpu   = points.detach().cpu()
#         cols_cpu  = colors.detach().cpu()
#         #for each candiate object get the point cloud unless there are too few points
#         for i in range(B):
#             pts = pts_cpu[i]          # (N,3)
#             cls = cols_cpu[i]         # (N,3)

#             # mask out void points
#             depths = pts[:, 2]
#             valid = (depths > config["min_depth"])# & (depths < config["max_depth"])
#             if debug:
#                 print(f"{valid.sum()=}")


#             if valid.sum() <  config["min_3d_points"]:
#                 continue
#             pts_valid = pts[valid]
#             cls_valid = cls[valid]

#             # build Open3D PointCloud
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(pts_valid.numpy())
#             pcd.colors = o3d.utility.Vector3dVector(cls_valid.numpy()/255)
#             pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])
#             # Apply statistical outlier removal to denoise the point cloud
#             if config["statistical_outlier_removal"]:
#                 pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=config["statistical_nb_neighbors"], std_ratio=config["statistical_std_ratio"])
#             if config["radius_outlier_removal"]:
#                 pcd, ind = pcd.remove_radius_outlier(nb_points=config["radius_nb_points"], radius=config["radius_radius"])
#             bbox = pcd.get_axis_aligned_bounding_box()
#             bbox.color = (1.0, 0.0, 0.0)  # red
#             full_pcds.append(pcd)
#             full_scores.append(scores[i])
#             full_bounding_boxes_3d.append(bbox)
#             full_masked_rgb.append(masked_rgb[i])
#             full_masked_depth.append(masked_depth[i])


#         # Perform NMS on the 3D bounding boxes
#         #print(f"Found {len(full_pcds)=} {len(full_bounding_boxes_3d)=}, {len(full_scores)=}, {len(full_masked_rgb)=}, {len(full_masked_depth)=}")
#         reduced_bounding_boxes_3d, reduced_scores, reduced_extras = nms(full_bounding_boxes_3d, full_scores, extra_data_lists=[full_pcds, full_masked_rgb, full_masked_depth], iou_threshold=config["iou_3d_reduction"], three_d=True)
#         reduced_pcds = reduced_extras[0]
#         reduced_rgb_masks = reduced_extras[1]
#         reduced_depth_masks = reduced_extras[2]

#         # Normalize the scores
#         reduced_scores = torch.tensor(reduced_scores)
#         #mu    = reduced_scores.mean()
#         #sigma = reduced_scores.std(unbiased=False)
#         #z = (reduced_scores - mu) / sigma
#         #reduced_scores = F.softmax(z, dim=0)
#         #reduced_scores = reduced_scores / reduced_scores.sum()
#         #reduced_scores = F.sigmoid(reduced_scores*config["sigmoid_gain"])
#         if debug:
#             print(f"   {len(full_pcds)=}, {len(full_scores)=}, {len(full_bounding_boxes_3d)=}, {len(full_masked_rgb)=}, {len(full_masked_depth)=}")
#             print(f"   {len(reduced_pcds)=}, {len(reduced_bounding_boxes_3d)=}, {reduced_scores.shape=}, {len(reduced_rgb_masks)=}, {len(reduced_depth_masks)=}")

#         return reduced_pcds, reduced_bounding_boxes_3d, reduced_scores, reduced_rgb_masks, reduced_depth_masks
#     def __str__(self):
#         return f"SAM2: {self.sam_predictor.model.device}"
#     def __repr__(self):
#         return self.__str__()



# def display_sam2(point_clouds, boxes, scores, window_prefix=""):
#     # 1) Initialize the GUI app (only once per program)
#     #print("Initializing GUI app...")
#     app = gui.Application.instance
#     app.initialize()

#     # 2) Create the O3DVisualizer window
#     #print("Creating O3DVisualizer window...")
#     vis = o3d.visualization.O3DVisualizer(
#         window_prefix + "PointClouds", 1024, 768
#     )
#     vis.show_settings = True

#     # 3) Add geometries by name
#     #print("Adding geometries...")
#     vis.add_geometry("CameraFrame",
#                      o3d.geometry.TriangleMesh.create_coordinate_frame(
#                          size=0.2, origin=[0,0,0]))
#     for idx, pc in enumerate(point_clouds):
#         vis.add_geometry(f"PointCloud{idx}", pc)
#     for idx, box in enumerate(boxes):
#         vis.add_geometry(f"Box{idx}", box)

#     # 4) Annotate each box’s center with its score
#     #print("Annotating boxes...")
#     for box, score in zip(boxes, scores):
#         center = box.get_center()  # AABB or mesh both support this
#         vis.add_3d_label(center, f"{float(score):.2f}")

#     # 5) Finalize and run
#     #print("Finalizing and running...")
#     vis.reset_camera_to_default()
#     app.add_window(vis)
#     app.run()

# def test_sam(rgb_img, depth_img, predictions, intrinsics, debug):
#     sam = SAM2_PC()
#     for query_object, canditates in predictions.items():
#         if debug:
#             print("\n\n")
#         point_clouds, boxes, scores, rgb_masks, depth_masks = sam.predict(rgb_img, depth_img, canditates["boxes"], canditates["scores"], intrinsics, debug=debug)
#         n = 5
#         fig, axes = plt.subplots(5, 2, figsize=(20, 10))
#         for i in range(min(n, len(point_clouds))):
#             axes[i, 0].imshow(rgb_masks[i])
#             axes[i, 1].imshow(depth_masks[i], cmap='gray')
#             axes[i, 0].set_title(f"{query_object} {i} Score:{scores[i]:.2f}")
#         fig.tight_layout()
#         fig.suptitle(f"{query_object} RGB and Depth Masks")
#         plt.show(block = False)
#         #print("begin display:")
#         display_sam2(point_clouds, boxes, scores, window_prefix=f"{query_object}")
#     plt.show(block = True)
#     return None


# if __name__=="__main__":
#     from RealsenseInterface import RealSenseCameraSubscriber

#     sub = RealSenseCameraSubscriber(
#         channel_name="realsense/Head",
#         InitChannelFactory=True
#     )
#     rgb_img, depth_img, Intrinsics, Extrinsics = None, None, None, None
#     while rgb_img is None or depth_img is None or Intrinsics is None or Extrinsics is None:
#         print("Waiting for RGB-D data...")
#         rgb_img, depth_img, Intrinsics, Extrinsics = sub.read(display=False)
#         #print(f"Received RGB-D data: {type(rgb_img)}, {type(depth_img)}, {type(Intrinsics)}, {type(Extrinsics)}")

#     I = {
#         "fx": Intrinsics[0, 0],
#         "fy": Intrinsics[1, 1],
#         "cx": Intrinsics[0, 2],
#         "cy": Intrinsics[1, 2]
#     }

#     print(f"\n\nTESTING OWL")
#     predictions = test_OWL(rgb_img, debug=False)

#     print(f"\n\nTESTING SAM")
#     test_sam(rgb_img, depth_img, predictions, I, debug=False)
