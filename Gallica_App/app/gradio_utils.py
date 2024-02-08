import os
from PIL import Image, ImageDraw, ImageFont
import json
import torch
import detectron2
import time
import cv2
import numpy as np
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
lP_weights_directory = weight_dir + 'LayoutParser/'
dt2_weights_directory = weight_dir + 'Detectron2/'

# Preset Default Configuration
default_array = [None, "Default", "", []]
default_detectron2 = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
default_layoutP = "model_final.pth"
# Json File with Ground Truth
json_path = gt_json_dir + "gallica_dataset_file.json"
with open(json_path) as file:
    json_data = json.load(file)
annotations = json_data["annotations"]
images = json_data["images"]
# Preset classes and default classes
default_classes = [
    'tampon',
    'écriture manuscrite',
    'écriture typographique',
    'photographie',
    'estampe'
]

classes = [
            'tampon',
            'écriture manuscrite',
            'écriture typographique',
            'photographie',
            'estampe',
            'décoration',
            'timbre',
            'dessin',
            'nothing'
        ]

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

def create_summary_message(selected_models, dt2_weight, lP_weight, class_selection):
    message = "Selected models: \n"
    if selected_models == [None, "", []]:
        message = "No Models used \n"
    # Handle Detectron2 model selection and configuration
    if "Detectron2" in selected_models:
        if dt2_weight in  [None, "Default", "", []] :
            message += "\n - Detectron2 with default configuration: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        else:
            message += f"\nDetectron2 with custom weight file: {dt2_weight}."
    
    # Handle LayoutParser model selection and configuration
    if "LayoutParser" in selected_models:
        if lP_weight in  [None, "Default", "", []] :
            message += "\n - LayoutParser with default configuration : model_final.pth"
        else:
            message += f"\n - LayoutParser with weight file: {lP_weight}."
    if "YOLOv8" in selected_models:
        message += "\nYolov8 Not implemented"
    # Handle class selection
    message +="\n"
    if class_selection == "Default Classes":
        used_classes = default_classes
        class_names = ", ".join(used_classes)
        message += f"\nUsing default classes: {len(default_classes)}\n{class_names}.\n"
    else:  # "All Classes"
        used_classes = classes
        class_names = ", ".join(used_classes)
        message += f"\nUsing all classes:{len(classes)}\n {class_names}.\n"
    return  message

def get_detectron2_paths(config):
    # Simplification: centralise la logique de configuration de Detectron2
    if config in default_array:
        return model_zoo.get_config_file(default_detectron2), model_zoo.get_checkpoint_url(default_detectron2)
    else:
        return model_zoo.get_config_file(default_detectron2), os.path.join(dt2_weights_directory, config)

def get_layoutparser_paths(weight):
    # Simplification: centralise la logique de configuration de LayoutParser
    if weight in default_array:
        return '../models/LayoutParser/config.yml', os.path.join(lP_weights_directory, default_layoutP)
    else:
        return '../models/LayoutParser/config.yml', os.path.join(lP_weights_directory, weight)

def initialize_predictors(selected_models, dt2_weight, lP_weight, num_class, detection_threshold):
    if isinstance(dt2_weight, list):
        dt2_weight = dt2_weight[0] if dt2_weight else "Default"
    predictors = {}
    
    for model in selected_models:
        if model == "Detectron2":
            config_path, weights_path = get_detectron2_paths(dt2_weight)
        elif model == "LayoutParser":
            config_path, weights_path = get_layoutparser_paths(lP_weight)
    
        if model in ["Detectron2", "LayoutParser"]: 
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
            predictors[model] = DefaultPredictor(cfg)
        else:
            #TODO YOLOv8
            continue
    return predictors

# main function for drop components
def process_drop(image_input, selected_models, dt2_weight, lP_weight, class_selection, detection_threshold):
    if class_selection == "Default Classes":
        num_class = len(default_classes)
    else: 
        num_class =  len(classes)
    # create information 
    message = create_summary_message(selected_models, dt2_weight, lP_weight, class_selection)
    # initialize models
    predictors = initialize_predictors(selected_models, dt2_weight, lP_weight, num_class, detection_threshold)
    # get Id
    file_id = extract_id_from_path(image_input)
    # get bounding boxes informations from json
    ground_truth_boxes = get_bounding_boxes(file_id, annotations)
    #Draw Bounding boxes    
    if ground_truth_boxes is None:
        message += "\nNo annotations found for this image.\n"
        gt_img, gt_box_info = None, None
    else:
        gt_img, gt_box_info = draw_bounding_boxes(image_input, ground_truth_boxes, use_percentage=True)
    
    img = Image.open(image_input)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    #Predict for each models
    predictions = []
    boxes_infos = ""
    if len(selected_models) == 0:
        message += "\nModels not used.\n"
        return message, img, gt_img, gt_box_info, predictions, boxes_infos
    boxes_infos += "Boxes Infos:\n"
    for model in selected_models:
        # Get model prediction boxes
        if model in ["Detectron2", "LayoutParser"]:
            prediction_boxes, time = predict_with_model(image_input, predictors[model])
            pred_img, info = draw_bounding_boxes(image_input, prediction_boxes)
            predictions.append((pred_img, model))
            boxes_infos += f"- {model} : \n  {info}\n"
        else:
            continue
    return message, img, gt_img, gt_box_info, predictions, boxes_infos

def predict_with_model(image_path, predictor):
    """
        From the path of image and a predictor such as detectron2 predict bounding boxes
         
        Output : array of bounding boxes, time 
    """
    # Measure the prediction time
    # Read the image with OpenCV
    im = cv2.imread(image_path)

    # Convert color from BGR to RGB (Detectron2 expects RGB)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    outputs = predictor(im)
    end_time = time.time()
        
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Extract the instances
    instances = outputs["instances"].to("cpu")
    
    # Check if any instances are detected
    if len(instances) > 0:
        # Extract bounding boxes, scores, and classes
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        labels = instances.pred_classes.numpy()
        # Filter predictions with a minimum confidence score
        predictions = []
        for box, num_label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            predictions.append({"bbox": {"x": xmin, "y": ymin, "width": xmax - xmin, "height": ymax - ymin}, 
                                "label": classes[num_label]})
        return predictions, elapsed_time
    else:
        print(f"No instances detected")
        return [], elapsed_time

def extract_id_from_path(file_path):
    """
        Input : image path
        Output : image name without extansion 
    """
    # Extract the file extension
    _, file_extension = os.path.splitext(file_path)
    # Check if the file is in an acceptable image format
    if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        print(file_name)
        return file_name
    else:
        return "Le fichier n'est pas une image"

def get_bounding_boxes(image_id, annotations):
    """
        Bounding boxes from Ground truth
    """
    for annotation in annotations:
        if annotation["id"] == image_id:
            return [{"bbox": result["bbox"], "label": result["label"][0] if result["label"] else "nothing"} for result in annotation["result"]]
    return None

def draw_bounding_boxes(image_path, boxes, use_percentage=False):
    """
    Draw bounding boxes over an image.
    
    :param image_input: PIL Image object
    :param boxes: A list of dictionaries, each containing the bounding box and label.
    :param use_percentage: If True, the bounding box coordinates are treated as percentages.
    """
    with Image.open(image_path) as img:
        # Convertir l'image en RVB si elle est en niveaux de gris
        if img.mode != 'RGB':
            img = img.convert('RGB')

        draw = ImageDraw.Draw(img)

        box_info = []  # To store information about the boxes
        
        for box in boxes:
            bbox = box["bbox"]
            label = box["label"]
            color = class_color_map.get(label, "black")
            x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            
            # Convert percentage to pixel values if required
            if use_percentage:
                x = x * img.width / 100
                y = y * img.height / 100
                width = width * img.width / 100
                height = height * img.height / 100
            
            # Draw the class name. Adjust the position as needed.
            try:
                font = ImageFont.truetype("../config/arial.ttf", 100)  # Adjust the font path as needed
            except IOError:
                font = ImageFont.load_default()
            text_position = (x + 5, y - 15)  # Adjust the position offset as per your requirement
            draw.text(text_position, label, fill=color, font=font)
            
            draw.rectangle([x, y, x + width, y + height], outline=color, width=10)
            box_info.append(f"Class: {label}, Coordinates: {x}, {y}, {width}, {height}")
    return img, box_info
