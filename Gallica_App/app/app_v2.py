import os
import gradio as gr
from detectron2 import model_zoo
from detectron2.config import get_cfg
from gradio_utils import list_weight_files, create_summary_message, classes, default_classes

models = ["Detectron2", "LayoutParser", "YOLOv8"]

weight_dir = '../weights/'
# Directories for the weight files
layoutparser_weights_directory = weight_dir + 'LayoutParser/'
detectron2_weights_directory = weight_dir + 'Detectron2/'

default_array = [None, "Default", "", []]
default_detectron2 = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
default_layoutP = "LP_weights.pth"
# Get the list of weight files for both models
layoutparser_weight_files = ["Default"] + list_weight_files(layoutparser_weights_directory)
detectron2_weight_files = ["Default"] + list_weight_files(detectron2_weights_directory)

# Define the main function that will use the selected options (updated to include Detectron2 weight path selection)
def process_images(selected_models, detectron2_config, layoutparser_weight, class_selection, message_information):
    message = create_summary_message(selected_models, detectron2_config, layoutparser_weight, class_selection)
    if isinstance(detectron2_config, list):
        detectron2_config = detectron2_config[0] if detectron2_config else "Default"
        
    if "Detectron2" in selected_models:
        cfg_detectron2 = get_cfg()
        if detectron2_config in default_array:
            cfg_detectron2.merge_from_file(model_zoo.get_config_file(default_detectron2))
            cfg_detectron2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(default_detectron2)
        else:
            cfg_detectron2.MODEL.WEIGHTS = os.path.join(detectron2_weights_directory, detectron2_config)
            
    if "LayoutParser" in selected_models:
        cfg_LayoutP = get_cfg()
        cfg_LayoutP.merge_from_file('../models/LayoutParser/config.yml')
        if layoutparser_weight in default_array:
            cfg_LayoutP.MODEL.WEIGHTS = os.path.join(layoutparser_weights_directory, default_layoutP)
        else:
            cfg_LayoutP.MODEL.WEIGHTS = os.path.join(layoutparser_weights_directory, layoutparser_weight)
    
    return message, message

# State is an class which can be used across blocks, tabs,...

with gr.Blocks(title="Gallica BNF Benchmark Visualisation Tool by EPITA", theme=gr.themes.Soft()) as demo:
    message_information = gr.State([])
    with gr.Tab("Set Evaluation"):
        with gr.Row():
            with gr.Column():
                # vairable correspond to a zone
                selected_models = gr.CheckboxGroup(choices=models, label="Select Model(s)", scale=2)
                layoutparser_weight = gr.Dropdown(choices=layoutparser_weight_files, label="LayoutParser Weight File")
                detectron2_config = gr.Dropdown(choices=detectron2_weight_files, label="Detectron2 Weight File")
                class_selection = gr.Radio(choices=["Default Classes", "All Classes"], label="Class Selection", value="Default Classes", scale=2)
                btn = gr.Button("Process Images", size='sm')
            with gr.Column():
                info = gr.Text(label="Information", scale=3)
                
        btn.click(process_images, inputs=[selected_models, detectron2_config, layoutparser_weight, class_selection, message_information], outputs=[info, message_information])
    with gr.Tab("Metrics"):
        metrics_output = gr.Text(label="Information", scale=3)
        # This button is used to refresh the metrics tab with the latest state information.
        refresh_btn = gr.Button("Refresh Metrics")
        def get_states(info):
            return info
        refresh_btn.click(get_states,inputs=message_information, outputs=metrics_output)

demo.launch()