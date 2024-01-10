import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import json
import torch
print(torch.__version__)
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
import os
import numpy as np
import cv2

# Load a configuration file
cfg = get_cfg()

# Specify the path to the config file for the pre-trained model
config_file = "C:\dev\pfeeApp\config.yml"
cfg.merge_from_file(config_file)

# Specify the path to the pre-trained model weights
#model_weights = "C:\dev\pfeeApp\model_final.pth"
#cfg.MODEL.WEIGHTS = model_weights

cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 1000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

# Set the device (CPU or GPU) for inference
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chemin vers le dossier des images et le fichier JSON
images_path = "image/"
json_file = "galica/gallicaimages_set1.json"

# Lire le fichier JSON
with open(json_file, "r", encoding="utf-8") as file:
    json_data = json.load(file)

annotations = json_data["annotations"]
images = json_data["images"]

def predict_with_model(image_path, predictor):

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
    
    # Read the image with OpenCV
    im = cv2.imread(image_path)

    # Convert color from BGR to RGB (Detectron2 expects RGB)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Perform prediction using Detectron2
    outputs = predictor(im)

    # Extract the instances
    instances = outputs["instances"].to("cpu")

    #print(instances.pred_boxes.tensor.tolist(),instances.pred_classes.tolist())

    # Check if any instances are detected
    if len(instances) > 0:
        # Extract bounding boxes, scores, and classes
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        labels = instances.pred_classes.numpy()

        # Filter predictions with a minimum confidence score
        predictions = []
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            predictions.append({"bbox": {"x": xmin, "y": ymin, "width": xmax - xmin, "height": ymax - ymin}, "label": classes[label]})

        return predictions
    else:
        print("No instances detected.")
        return []

def get_bounding_boxes(image_id):
    for annotation in annotations:
        if annotation["id"] == image_id:
            return [{"bbox": result["bbox"], "label": result["label"][0] if result["label"] else "nothing"} for result in annotation["result"]]
    return None

# Fonction pour dessiner des bounding boxes sur une image
def draw_bounding_boxes(image_path, boxes,use_percentage = False):
    with Image.open(image_path) as img:
        # Convertir l'image en RVB si elle est en niveaux de gris
        if img.mode != 'RGB':
            img = img.convert('RGB')

        draw = ImageDraw.Draw(img)

        # Carte des couleurs pour les classes
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


        box_info = []  # Pour stocker les informations sur les boîtes

        for box in boxes:
                bbox = box["bbox"]
                label = box["label"]
                color = class_color_map.get(label, "black")
                x = bbox["x"]
                y = bbox["y"]
                width = bbox["width"]
                height = bbox["height"]

                 # Convert percentage to pixel values if required
                if use_percentage:
                    x = x * img.width / 100
                    y = y * img.height / 100
                    width = width * img.width / 100
                    height = height * img.height / 100

                # Draw the class name. Adjust the position as needed.
                font = ImageFont.truetype("arial.ttf", 15)
                text_position = (x + 5, y - 15)  # Adjust the position offset as per your requirement
                draw.text(text_position, label, fill=color, font=font)

                draw.rectangle([x, y, x + width, y + height], outline=color, width=3)
                box_info.append(f"Class: {label}, Coordinates: {x}, {y}, {width}, {height}")

        return img, box_info
      
def combined_image_processing(image_id, model):
    selected_image_file = f"{image_id}.jpg"
    image_path = f"{images_path}{selected_image_file}"

    # Get ground truth boxes
    ground_truth_boxes = get_bounding_boxes(image_id)
    if ground_truth_boxes is None:
        return "No annotations found for this image."

    # Draw ground truth bounding boxes
    gt_img, gt_box_info = draw_bounding_boxes(image_path, ground_truth_boxes, use_percentage=True)

    # If model predictions are requested
    if model != "None":
        model_weights = "C:\dev\pfeeApp\models\\" + model + ".pth"
        cfg.MODEL.WEIGHTS = model_weights
        predictor = DefaultPredictor(cfg)
        # Get model prediction boxes
        prediction_boxes = predict_with_model(image_path, predictor)
        # Draw prediction bounding boxes
        pred_img, pred_box_info = draw_bounding_boxes(image_path, prediction_boxes)
        return gt_img, gt_box_info, pred_img, pred_box_info

    # If model predictions are not requested, return only ground truth
    return gt_img, gt_box_info, None, "No model predictions requested."


# Interface Gradio
image_id_list = [image["id"] for image in images]

#list models in models folder without the extension
models_list = ["None"] + [model.split(".")[0] for model in os.listdir("models")]
print(models_list)

iface = gr.Interface(
    fn=combined_image_processing,
    inputs=[
        gr.Dropdown(choices=image_id_list),
        gr.Dropdown(choices=models_list, label="Model")
    ],
    outputs=[
        gr.Image(type="pil", label="Ground Truth Image"),
        gr.Textbox(label="Ground Truth Box Info"),
        gr.Image(type="pil", label="Prediction Image"),
        gr.Textbox(label="Prediction Box Info")
    ],
    title="Galica Image Annotation"
)

iface.launch()