Grasp VLM Prediction  
The goal is to see how vlms can produce bounding boxes for finger placements on tools. 
PLEASE NOTE THAT GENERALLY INDEX FINGER OBJECTS ARE MORE DIFFICULT BECAUSE THEIR LABELLED BBOX IN THE DATASET ARE MUCH SMALLER.
Results so far: 
![IoUs for All Models(Including Owl VIT and YOLO UniOW)](src/iou_boxplt_all_models.png)
![IoUs for Vlms](src/iou_boxplt_t.png)  
![IoUs for Owl VIT and YOLO UniOW](src/iou_boxplt_owl_yolo_experiment.png)  
For owl, I experimented with prompting nouns, specifically with single word prompts vs. descriptive prompts. I later redid Owl to use the same prompts that I found worked best with YOLO, which fundamentally is slightly different because it uses wildcard embeddings and works by identifying bound boxes and then classifying.

The dataset has 200 images and there are two main classes: index finger objects and four finger grasp objects.  
For index finger objects, the agent outputs a bbox for the index finger for grasping. For four finger objects, the agent outputs a bbox for the four fingers that would wrap around the object. Within these two classes, there are many tools.
The tools include:  
Pistol Grip Objects:  
    drill, weed wackers, glue gun, circular saw, nailgun.  
Handle Objects:  
    screwdriver, wrench, soldering tool, allen key, hammer.  
Annotations for Pistol Grip Objects:  
    bounding box for index finger placement.  
    pixel coordinates (x,y) for perfect placement of index finger.  
Annotations for Handle Objects:  
    bounding box for anywhere that is valid to place all 4 fingers.  
    pixel coordinates (x,y) for placements of all 4 fingers.  

