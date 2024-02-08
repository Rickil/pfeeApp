import os
import gradio as gr
from gradio_utils import list_weight_files, create_summary_message, process_drop, initialize_predictors, process_gallery
from gradio_utils import classes, default_classes, dt2_weights_directory, lP_weights_directory

# Available models for list
models = ["Detectron2", "LayoutParser", "YOLOv8"]
# Get the list of weight files for both models

def update_dropdowns():
    layoutparser_weight_files = ["Default"] + list_weight_files(lP_weights_directory)
    detectron2_weight_files = ["Default"] + list_weight_files(dt2_weights_directory)
    LPdropDown = gr.Dropdown(choices=layoutparser_weight_files,  label="LayoutParser Weight File")
    DT2dropDown = gr.Dropdown(choices=detectron2_weight_files , label="Detectron2 Weight File")
    return LPdropDown,  DT2dropDown
    


def setup_model_selection_interface():
    selected_models = gr.CheckboxGroup(choices=models, label="Select Model(s)", scale=2)
    
    lp_weights_files = ["Default"] + list_weight_files(lP_weights_directory)
    dt2_weight_files = ["Default"] + list_weight_files(dt2_weights_directory)
    layoutparser_weight = gr.Dropdown(choices=lp_weights_files, label="LayoutParser Weight File", allow_custom_value=True)
    detectron2_weight = gr.Dropdown(choices=dt2_weight_files, label="Detectron2 Weight File", allow_custom_value=True)
    ref_button = gr.Button("Refresh list", size='sm')
                
    class_selection = gr.Radio(choices=["Default Classes", "All Classes"], label="Class Selection", value="Default Classes", scale=2)
    threshold_input = gr.Number(label="Detection Threshold", value=0.8, step=0.01, minimum=0.0, maximum=1.0, interactive=True)
    
    # Returning all components to include in the UI
    return (selected_models, layoutparser_weight, detectron2_weight, ref_button, class_selection, threshold_input)
# State is an class which can be used across blocks, tabs,...
with gr.Blocks(title="Gallica BNF Benchmark Visualisation Tool by EPITA", theme=gr.themes.Soft()) as demo:
    ground_truths = gr.State([])
    predictions = gr.State([])
    message = gr.State([])
    metrics = gr.State([])
    
    with gr.Tab("Set Evaluation"):
        with gr.Row():
            with gr.Column():
                # Instanciate gradio objects, TODO: simplify instaciation, regroupe weight obteained if possible for more visbility
                selected_mds, lP_weight, dt2_weight, ref_btn, selected_cls, threshold_input = setup_model_selection_interface()
                ref_btn.click(
                    fn=update_dropdowns,
                    inputs=[],
                    outputs=[lP_weight, dt2_weight]
                )
                btn_process_images = gr.Button("Process Images", size='sm')
                     
            with gr.Column():
                info = gr.Text(label="Information", scale=3)  
                
            btn_process_images.click(process_gallery, 
                                    inputs=[selected_mds, dt2_weight, lP_weight, selected_cls, threshold_input], 
                                    outputs=[info, ground_truths, predictions])  
    with gr.Tab("Metrics"):
        metrics_output = gr.Text(label="Information", scale=3)
        # This button is used to refresh the metrics tab with the latest state information.
        refresh_btn = gr.Button("Refresh Metrics")
        def get_data():
            return metrics
        refresh_btn.click(get_data, outputs=metrics_output)
        # add an upload of the report HTML static 
    with gr.Tab("Gallery"):
        # Galleries of each Models Maybe add a DropDown to select display
        with gr.Column():
            gt = gr.Gallery(label="GroundTruth")
            pred = gr.Gallery(label="Predictions")
            refresh_btn = gr.Button("Refresh Metrics")
            def get_gt_pred():
                return ground_truths, predictions
            refresh_btn.click(get_gt_pred, outputs=[gt, pred]) 
    with gr.Tab("Evaluate an Image"):
        with gr.Row():
            with gr.Column():
                selected_mds, lP_weight, dt2_weight, ref_btn, selected_cls, threshold_input = setup_model_selection_interface()
                ref_btn.click(
                    fn=update_dropdowns,
                    inputs=[],
                    outputs=[lP_weight, dt2_weight]
                )
                btn_process_drop = gr.Button("Process Images", size='sm')
                     
            with gr.Column():
                info = gr.Text(label="Information", scale=3)  
                input_file = gr.File(type="filepath")
        with gr.Row():
            with gr.Row():
                im = gr.Image(label="Drop Image", scale= 2, mirror_webcam=False, interactive=False)
                gt_img = gr.Image(label="Ground Truth", scale=2, interactive=False)
                gt_info = gr.Textbox(label="Ground Truth Box Info", interactive=False)
        with gr.Row():
            pred_output = gr.Gallery(label="Predictions", scale=4, interactive=True, show_label=True)
            box_info = gr.Textbox(label="Prediction Box Info", interactive=False)
            
        btn_process_drop.click(process_drop, 
                                inputs=[input_file, selected_mds, dt2_weight, lP_weight, selected_cls, threshold_input], 
                                outputs=[info, im, gt_img, gt_info, pred_output, box_info])
demo.launch()