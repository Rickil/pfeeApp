import os
# Preset classes and default classes
classes = {
    'tampon': 0,
    'écriture manuscrite': 1,
    'écriture typographique': 2,
    'photographie': 3,
    'estampe': 4,
    'décoration': 5,
    'timbre': 6,
    'dessin': 7,
    'nothing': 8,
}

default_classes = {
    'tampon': 0,
    'écriture manuscrite': 1,
    'écriture typographique': 2,
    'photographie': 3,
    'estampe': 4
}

def list_weight_files(directory):
    """
        Function to list weight files in a directory
    """
    # Ensure the directory exists to avoid errors
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    # List all files in the directory 
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)) and file.endswith(".pth")]

def create_summary_message(selected_models, detectron2_config, layoutparser_weight, class_selection):
    message = "Selected models: \n"
    if selected_models == [None, "", []]:
        message = "No Models used \n"
    # Handle Detectron2 model selection and configuration
    if "Detectron2" in selected_models:
        if detectron2_config in  [None, "Default", "", []] :
            message += "\n - Detectron2 with default configuration: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        else:
            message += f"\nDetectron2 with custom weight file: {detectron2_config}."
    
    # Handle LayoutParser model selection and configuration
    if "LayoutParser" in selected_models:
        if layoutparser_weight in  [None, "Default", "", []] :
            message += "\n - LayoutParser with default configuration : LP_weights.pth"
        else:
            message += f"\nLayoutParser with weight file: {layoutparser_weight}."
    if "YOLOv8" in selected_models:
        message += "\nYolov8 Not implemented"
    # Handle class selection
    message +="\n"
    if class_selection == "Default Classes":
        used_classes = default_classes
        class_names = ", ".join(used_classes.keys())
        message += f"\nUsing default classes: \n {class_names}."
    else:  # "All Classes"
        used_classes = classes
        class_names = ", ".join(used_classes.keys())
        message += f"\nUsing all classes: \n {class_names}."
    return  message