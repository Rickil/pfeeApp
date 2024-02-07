import gradio as gr
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
import cv2
import numpy as np
from PIL import Image
import os
images_path = "../data/images/"
ground_truth_path = "../data/annotations/gallicaimages_set1.json"
config_file = "../config/config.yml"
models_folder = "../models/"

# Classes
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

def init_detectron2_model(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    if weights_path == "Model Zoo Default":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = weights_path  # Path to the custom weights file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust threshold as needed
    return DefaultPredictor(cfg)

# Function to process multiple uploaded images with Detectron2 using the selected weights
def process_images_gallery(image_files, model_choice, weights_choice):
    processed_images = []
    ground_truth_images = []
    
    if model_choice == "Detectron2":
        predictor = init_detectron2_model(weights_choice)
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file.name)
            im = cv2.imread(image_path)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = v.get_image()[:, :, ::-1]
            result_pil = Image.fromarray(result_image.astype(np.uint8))
            processed_images.append(result_pil)
    else:
        for image_file in image_files:
            processed_images.append(Image.open(image_file.name))
    return processed_images

weight_files = ["Model Zoo Default"] + [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.pth')]


with gr.Blocks() as demo:
    with gr.Tab("Upload and Process"):
        weights_dropdown = gr.Dropdown(choices=weight_files, label="Select Weights", value="Model Zoo Default")
        image_input = gr.File(label="Upload Image", type="filepath")
        process_button = gr.Button("Process Images")
    
    with gr.Tab("Gallery View"):
        gallery_processed = gr.Gallery(label="Processed Images")
        gallery_ground_truth = gr.Gallery(label="Ground Truth Images")

    process_button.click(
        fn=process_images_gallery,
        inputs=[image_input, weights_dropdown],
        outputs=[gallery_processed, gallery_ground_truth]
    )

demo.launch()