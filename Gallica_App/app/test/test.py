import gradio as gr
import json

images_path = "../data/images/"
json_file = "../data/annotations/gallicaimages_set1.json"

category_dict = {
    'tampon': 0,
    'écriture manuscrite': 1,
    'écriture typographique': 2,
    'photographie': 3,
    'estampe': 4,
    #  'décoration':5,
    #  'timbre':6,
    #  'dessin':7,
}
models = ["Detectron2", "LayoutParser", "Yolov8 (Not implemented)"]

with open(json_file, "r", encoding="utf-8") as file:
    json_data = json.load(file)

annotations = json_data["annotations"]
images = json_data["images"]

def process_selection(selected_models, detectron2_config, layoutparser_weight, class_selection):
    general_warning = ""
    # Check if Yolov8 is selected and set a warning
    if "Yolov8 (Not implemented)" in selected_models:
        selected_models = [model for model in selected_models if model != "Yolov8 (Not implemented)"]  # Remove Yolov8 from output
    if not selected_models:  # If no model is selected after removing Yolov8
        general_warning = "Warning: No model selected."
    selected_models_str = ", ".join(selected_models) if selected_models else "None"
    return f"{', '.join(selected_classes)}", f"{selected_models_str}", general_warning


with gr.Blocks() as demo:
    with gr.Row():
        checkbox_group_classes = gr.CheckboxGroup(choices=list(category_dict.keys()), label="Select Classes", value=list(category_dict.keys())[:5])
    
        
        checkbox_group_models = gr.CheckboxGroup(choices=models, label="Select Model", value=[])
        warning_label = gr.Textbox(label="Status")
    with gr.Column():
        output_classes = gr.Textbox(label="Classes Selection")
        output_model = gr.Textbox(label="Model Selection")
       
        submit_button = gr.Button("Submit", size='sm')

    submit_button.click(
        process_selection, 
        inputs=[checkbox_group_classes, checkbox_group_models], 
        outputs=[output_classes, output_model, warning_label]
    )

demo.launch()
