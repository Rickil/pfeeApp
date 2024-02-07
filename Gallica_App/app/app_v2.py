import os
import gradio as gr
from gradio_utils import list_weight_files, create_summary_message, \
                classes, default_classes, initialize_predictors, layoutparser_weights_directory, detectron2_weights_directory

# Available models for list
models = ["Detectron2", "LayoutParser", "YOLOv8"]
# Get the list of weight files for both models

def update_dropdowns():
    layoutparser_weight_files = ["Default"] + list_weight_files(layoutparser_weights_directory)
    detectron2_weight_files = ["Default"] + list_weight_files(detectron2_weights_directory)
    LPdropDown = gr.Dropdown(choices=layoutparser_weight_files,  label="LayoutParser Weight File")
    DT2dropDown = gr.Dropdown(choices=detectron2_weight_files , label="Detectron2 Weight File")
    return LPdropDown,  DT2dropDown
    
# Define the main function that will use the selected options (updated to include Detectron2 weight path selection) in Gradio
def process_images(selected_models, detectron2_weight,
                   layoutparser_weight, class_selection, detection_threshold):
    #create un message to display
    message = create_summary_message(selected_models, detectron2_weight, 
                                     layoutparser_weight, class_selection)
    try:
        # Setup predictors
        predictors = initialize_predictors(selected_models, detectron2_weight, layoutparser_weight, detection_threshold, class_selection)
        # If successful, you could append a success message or details about the predictors
        message += "\nPredictors initialized successfully."
    except Exception as e:
        # Append error message to the message string
        message += f"\nError initializing predictors: {str(e)}"
        
    return message

def setup_model_selection_interface():
    selected_models = gr.CheckboxGroup(choices=models, label="Select Model(s)", scale=2)
    
    lp_weights_files = ["Default"] + list_weight_files(layoutparser_weights_directory)
    dt2_weight_files = ["Default"] + list_weight_files(detectron2_weights_directory)
    layoutparser_weight = gr.Dropdown(choices=lp_weights_files, label="LayoutParser Weight File", allow_custom_value=True)
    detectron2_weight = gr.Dropdown(choices=dt2_weight_files, label="Detectron2 Weight File", allow_custom_value=True)
    ref_button = gr.Button("Refresh list", size='sm')
                
    class_selection = gr.Radio(choices=["Default Classes", "All Classes"], label="Class Selection", value="Default Classes", scale=2)
    threshold_input = gr.Number(label="Detection Threshold", value=0.8, step=0.01, minimum=0.0, maximum=1.0, interactive=True)
    
    # Returning all components to include in the UI
    return (selected_models, layoutparser_weight, detectron2_weight, ref_button, class_selection, threshold_input)
# State is an class which can be used across blocks, tabs,...
model_selection_components = setup_model_selection_interface()
with gr.Blocks(title="Gallica BNF Benchmark Visualisation Tool by EPITA", theme=gr.themes.Soft()) as demo:
    message_information = gr.State([])
    with gr.Tab("Set Evaluation"):
        with gr.Row():
            with gr.Column():
                
                selected_models, layoutparser_weight, detectron2_weight, ref_button, class_selection, threshold_input = setup_model_selection_interface()
                ref_button.click(
                    fn=update_dropdowns,
                    inputs=[],
                    outputs=[layoutparser_weight, detectron2_weight]
                )
                btn_process_images = gr.Button("Process Images", size='sm')
                     
            with gr.Column():
                info = gr.Text(label="Information", scale=3)  
                
        btn_process_images.click(process_images, 
                                 inputs=[selected_models, detectron2_weight, layoutparser_weight, class_selection, threshold_input], 
                                 outputs=[info])  
    with gr.Tab("Metrics"):
        metrics_output = gr.Text(label="Information", scale=3)
        # This button is used to refresh the metrics tab with the latest state information.
        refresh_btn = gr.Button("Refresh Metrics")
        def get_states(info):
            return info
        refresh_btn.click(get_states,inputs=message_information, outputs=metrics_output)
        # add an upload of the report HTML static 
    with gr.Tab("Gallery"):
        # Galleries of each Models Maybe add a DropDown to select display
        with gr.Column():
             gt = gr.Gallery(label="GroundTruth")
             det_gl = gr.Gallery(label="Detectron2")
             lay_gl = gr.Gallery(label="LayoutParser")
    with gr.Tab("Evaluate an Image"):
        #TODO SAME AS above but not with ground truth , random Image
         with gr.Row():
            with gr.Column():
                
                selected_models, layoutparser_weight, detectron2_weight, ref_button, class_selection, threshold_input = setup_model_selection_interface()
                ref_button.click(
                    fn=update_dropdowns,
                    inputs=[],
                    outputs=[layoutparser_weight, detectron2_weight]
                )
                btn_process_images = gr.Button("Process Images", size='sm')
                     
            with gr.Column():
                info = gr.Text(label="Information", scale=1)  
                image_input = gr.Image(label="Drop Image", scale= 2)
                image_output = gr.Image()
            
            btn_process_images.click(process_images, 
                                 inputs=[selected_models, detectron2_weight, layoutparser_weight, class_selection, threshold_input], 
                                 outputs=[info])  
    
demo.launch()