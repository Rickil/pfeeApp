from PIL import Image, ImageDraw, ImageFont
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
import os
import cv2

def predict_with_model(classes, image_path, predictor):
    """
        From the path of image and a predictor such as detectron2 predict bounding boxes
         
        Output : array of bounding boxes
    """
    # Read the image with OpenCV
    im = cv2.imread(image_path)
    # Convert color from BGR to RGB (Detectron2 expects RGB)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Perform prediction using Detectron2
    outputs = predictor(im)

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
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            predictions.append({"bbox": {"x": xmin, "y": ymin, "width": xmax - xmin, "height": ymax - ymin}, "label": classes[label]})
        return predictions
    else:
        print("No instances detected.")
        return []

def get_Ground_truth_bd_boxes(image_id, annotations):
    """
        Input : image id and corresponding annotations
        Output : array of bounding boxes 
    """
    for annotation in annotations:
        if annotation["id"] == image_id:
            return [{"bbox": result["bbox"], "label": result["label"][0] if result["label"] else "nothing"} for result in annotation["result"]]
    return None

# Fonction pour dessiner des bounding boxes sur une image
def draw_bounding_boxes(image_path, boxes, use_percentage = False):
    """
        Draw bounding boxes over an image
    """
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
            'décoration': "cyan",
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
                font = ImageFont.truetype("../config/arial.ttf", 15)
                text_position = (x + 5, y - 15)  # Adjust the position offset as per your requirement
                draw.text(text_position, label, fill=color, font=font)

                draw.rectangle([x, y, x + width, y + height], outline=color, width=3)
                box_info.append(f"Class: {label}, Coordinates: {x}, {y}, {width}, {height}")

        return img, box_info
