import json

data_dir = '../data/' 
weight_dir = '../weights/'
gt_json_dir = data_dir + 'annotations/'
img_dir = data_dir + 'images/'

# Directories for the weight files
lP_weights_directory = weight_dir + 'LayoutParser/'
dt2_weights_directory = weight_dir + 'Detectron2/'

# Preset Default Configuration
default_array = [None, "Default", "", []]
default_detectron2 = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
default_layoutP = "model_final.pth"
# Json File with Ground Truth
json_path = gt_json_dir + "gallica_dataset_file.json"
with open(json_path) as file:
    json_data = json.load(file)
annotations = json_data["annotations"]
images = json_data["images"]
image_id = "005c23c0-8b19-4647-8717-b01760087598"

def extract_id_from_path(file_path):
    """
        Input : image path
        Output : image name without extansion 
    """
    # Extract the file extension
    _, file_extension = os.path.splitext(file_path)
    # Check if the file is in an acceptable image format
    if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        
        return file_name
    else:
        return "Le fichier n'est pas une image"

file_name = extract_id_from_path(file_path)
for annotation in annotations:
    if annotation["id"] == image_id:
        print("ok")