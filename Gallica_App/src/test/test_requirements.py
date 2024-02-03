import gradio as gr
from PIL import Image
import torch
import torchvision
import detectron2
import numpy as np
import cv2
import ultralytics

def check_versions():
    versions = {
        "Gradio": gr.__version__,
        "Pillow": Image.__package__,
        "Torch": torch.__version__,
        "Torchvision": torchvision.__version__,
        "Detectron2": detectron2.__version__,
        "NumPy": np.__version__,
        "OpenCV": cv2.__version__, 
        "Ultralytics":ultralytics.__version__
        
    }
    return "\n".join([f"{lib}: {ver}" for lib, ver in versions.items()])

# Gradio interface to display versions
iface = gr.Interface(fn=check_versions, inputs=[], outputs="text")
iface.launch(server_name='0.0.0.0', server_port=7860)
