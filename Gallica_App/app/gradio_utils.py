import os
from PIL import Image, ImageDraw, ImageFont
import json
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo

data_dir = '../data/' 
weight_dir = '../weights/'
gt_json_dir = data_dir + 'annotations/'
img_dir = data_dir + 'images/'

# Directories for the weight files
layoutparser_weights_directory = weight_dir + 'LayoutParser/'
detectron2_weights_directory = weight_dir + 'Detectron2/'

# Preset Default Configuration
default_array = [None, "Default", "", []]
default_detectron2 = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
default_layoutP = "model_final.pth"
# Json File with Ground Truth
json_file = gt_json_dir + "gallicaimages_set1.json"

# Preset classes and default classes
classes = {
    'tampon': 0,
    'écriture manuscrite': 1,
    'écriture typographique': 2,
    'photographie': 3,
    'estampe': 4,
    'décoration': 5,
    'timbre': 6,
    'dessin': 7,
    'nothing': 8,
}

default_classes = {
    'tampon': 0,
    'écriture manuscrite': 1,
    'écriture typographique': 2,
    'photographie': 3,
    'estampe': 4
}

class_color_map = {
            'nothing': "black",
            'tampon': "blue",
            'écriture manuscrite': "green",
            'écriture typographique': "yellow",
            'photographie': "purple",
            'estampe': "orange",
            'décoration': "pink",
            'timbre': "red",
            'dessin': "brown"
        }
def update_dropdown(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    # Update the dropdown options dynamically
    dropdown.update(choices=files)
    return "Dropdown updated!"


def list_weight_files(directory):
    """
        Function to list weight files in a directory
    """
    # Ensure the directory exists to avoid errors
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    # List all files in the directory 
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and file.endswith(".pth")]

def create_summary_message(selected_models, detectron2_config, layoutparser_weight, class_selection):
    message = "Selected models: \n"
    if selected_models == [None, "", []]:
        message = "No Models used \n"
    # Handle Detectron2 model selection and configuration
    if "Detectron2" in selected_models:
        if detectron2_config in  [None, "Default", "", []] :
            message += "\n - Detectron2 with default configuration: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        else:
            message += f"\nDetectron2 with custom weight file: {detectron2_config}."
    
    # Handle LayoutParser model selection and configuration
    if "LayoutParser" in selected_models:
        if layoutparser_weight in  [None, "Default", "", []] :
            message += "\n - LayoutParser with default configuration : model_final.pth"
        else:
            message += f"\n - LayoutParser with weight file: {layoutparser_weight}."
    if "YOLOv8" in selected_models:
        message += "\nYolov8 Not implemented"
    # Handle class selection
    message +="\n"
    if class_selection == "Default Classes":
        used_classes = default_classes
        class_names = ", ".join(used_classes.keys())
        message += f"\nUsing default classes: {len(default_classes)}\n{class_names}.\n"
    else:  # "All Classes"
        used_classes = classes
        class_names = ", ".join(used_classes.keys())
        message += f"\nUsing all classes:{len(default_classes)}\n {class_names}.\n"
    return  message

def get_detectron2_paths(config):
    # Simplification: centralise la logique de configuration de Detectron2
    if config in default_array:
        return model_zoo.get_config_file(default_detectron2), model_zoo.get_checkpoint_url(default_detectron2)
    else:
        return os.path.join(detectron2_weights_directory, config), os.path.join(detectron2_weights_directory, config)

def get_layoutparser_paths(weight):
    # Simplification: centralise la logique de configuration de LayoutParser
    if weight in default_array:
        return '../models/LayoutParser/config.yml', os.path.join(layoutparser_weights_directory, default_layoutP)
    else:
        return '../models/LayoutParser/config.yml', os.path.join(layoutparser_weights_directory, weight)

def initialize_predictors(selected_models, detectron2_weight, layoutparser_weight, detection_threshold, class_selection):
    if isinstance(detectron2_weight, list):
        detectron2_weight = detectron2_weight[0] if detectron2_weight else "Default"
    predictors = {}
    for model in selected_models:
        cfg = get_cfg()
        if model == "Detectron2":
            config_path, weights_path = get_detectron2_paths(detectron2_weight)
        elif model == "LayoutParser":
            config_path, weights_path = get_layoutparser_paths(layoutparser_weight)
        else:
            continue
        
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold
        predictors[model] = DefaultPredictor(cfg)
    return predictors
    