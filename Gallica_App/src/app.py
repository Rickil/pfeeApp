import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import json
import torch
print(torch.__version__)
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from model_utils import predict_with_model, draw_bounding_boxes, get_Ground_truth_bd_boxes
import os
import cv2

# Image Folder and JSON paths
images_path = "../data/images/"
json_file = "../data/annotations/gallicaimages_set1.json"
# Specify the path to the config file for the pre-trained model
config_file = "../config/config.yml"
models_folder = "../results/models/"

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

# Reading JSON 
with open(json_file, "r", encoding="utf-8") as file:
    json_data = json.load(file)

annotations = json_data["annotations"]
images = json_data["images"]

# Detectron2 Parameters
cfg = get_cfg()
# Load a configuration file
cfg.merge_from_file(config_file)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 1000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
# Set the device (CPU or GPU) for inference
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def combined_image_processing(image_id, model):
    """
        Function for Gradio app
        Process an Image according to a model or the default model
        
        Input image_id and a model
    """
    selected_image_file = f"{image_id}.jpg"
    image_path = f"{images_path}{selected_image_file}"

    # Get ground truth boxes
    ground_truth_boxes = get_Ground_truth_bd_boxes(image_id, annotations)
    if ground_truth_boxes is None:
        return "No annotations found for this image."

    # Draw ground truth bounding boxes
    gt_img, gt_box_info = draw_bounding_boxes(image_path, ground_truth_boxes, use_percentage=True)

    # If model predictions are requested
    if model != "None":
        model_weights = models_folder + model + ".pth"
        cfg.MODEL.WEIGHTS = model_weights
        predictor = DefaultPredictor(cfg)
        # Get model prediction boxes
        prediction_boxes = predict_with_model(classes, image_path, predictor)
        # Draw prediction bounding boxes
        pred_img, pred_box_info = draw_bounding_boxes(image_path,  prediction_boxes)
        return gt_img, gt_box_info, pred_img, pred_box_info

    # If model predictions are not requested, return only ground truth
    return gt_img, gt_box_info, None, "No model predictions requested."
# Interface Gradio
image_id_list = [image["id"] for image in images]

#list models in models folder without the extension
models_list = ["None"] + [model.split(".")[0] for model in os.listdir(models_folder)]

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