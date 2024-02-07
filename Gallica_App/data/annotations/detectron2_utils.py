from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from io import BytesIO
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import PIL.Image
import requests
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def  select_detectron2_model():
    """
        Set Up Detectron2 predictor
        Input : 
    """
    cfg = get_cfg()

    if  selected_model == "Detectron2" and :
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    if selected_model == "LayoutParser":
        cfg.merge_from_file(modeldata_path + "config.yml")
        cfg.MODEL.WEIGHTS = modeldata_path + "Layout_parser_weights.pth"    
    else :
        raise ValueError("Invalid model selected!")