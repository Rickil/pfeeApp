# Functions from GallicaTraining
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader

import json
import os
import numpy as np

def get_my_dataset_dicts(json_file, category_dict, image_dir):
    """
        Convert a Label Studio json to COCO dataset json
    """
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)

    dataset_dicts = []
    for idx, image in enumerate(data['images']):
        record = {}
        # Create file path from local directory and image id
        filename = os.path.join(image_dir, image['id'] + '.jpg')

        # Assign image properties
        record["file_name"] = filename
        record["height"] = image["height"] 
        record["width"] = image["width"]
        record["image_id"] = image['id']

        annotations = []
        # Assign annotations to image
        for ann in data["annotations"]:
            for result in ann["result"]:
                if ann["id"] == image["id"] and type(result["label"]) is not int and \
                    result["label"][0] in category_dict.keys():
                # Create a new dict for each annotation in the image
                    x = int(result['bbox']['x'] / 100.0 * image["width"])
                    y = int(result['bbox']['y'] / 100.0 * image["height"])
                    width = int(result['bbox']['width'] / 100.0 * image["width"])
                    height = int(result['bbox']['height'] / 100.0 * image["height"])
                    obj = {
                        "bbox": [x, y, width, height],
                        "bbox_mode": BoxMode.XYWH_ABS,  # as your bounding box coordinates are in absolute format
                        "category_id": category_dict[result["label"][0]],  # map your label name to its corresponding id
                    }
                    annotations.append(obj)
        record["annotations"] = annotations
        dataset_dicts.append(record)
    return dataset_dicts
    
def remove_all_datasets():
    registered_datasets = list(DatasetCatalog.list())
    for dataset_name in registered_datasets:
        DatasetCatalog.remove(dataset_name)

def split_data(json_file, category_dict, image_dir, split='train'):
    # Load json file
    """
        Divide original dataset in train, val, test according to a certain class
    """
    with open(json_file) as f:
        data = json.load(f)

    classes_count = np.zeros(len(category_dict))
    # Convert to Detectron2 format
    dataset_dicts = get_my_dataset_dicts(json_file, category_dict, image_dir)
    print(len(dataset_dicts))
    
    #get total number of each classes
    for data in dataset_dicts:        
        for annotation in data["annotations"]:
            classes_count[annotation["category_id"]] += 1
    
    data_train = []
    data_val = []
    data_test = []
    data_lost = []
    nb_image = 0
    count = np.zeros(len(category_dict))
    classes_count_sorted = np.argsort(classes_count)
    for data in dataset_dicts:
        isLost = True
        for annotation in data["annotations"]:
            v = False
            for i in range(len(category_dict)):
                if count[annotation["category_id"]] < classes_count[classes_count_sorted[i]]*0.83:
                    data_train.append(data)
                    for annotation in data["annotations"]:
                        count[annotation["category_id"]] += 1
                    nb_image+=1
                    v=True
                    break
                elif count[annotation["category_id"]] < classes_count[classes_count_sorted[i]]*0.95:
                    data_val.append(data)
                    for annotation in data["annotations"]:
                        count[annotation["category_id"]] += 1
                    nb_image+=1
                    v=True
                    break
                elif count[annotation["category_id"]] < classes_count[classes_count_sorted[i]]:
                    data_test.append(data)
                    for annotation in data["annotations"]:
                        count[annotation["category_id"]] += 1
                    nb_image+=1
                    v=True
                    break
            if v:
                isLost = False
                break
        if isLost:
            data_lost.append(data)
            
    #append lost data to test data
    for data in data_lost:
        data_test.append(data)
                        
    print("count: ", count)
    print("classes_count: ", classes_count)
    print("nb_image: ", nb_image)
    print("data_lost", len(data_lost))
    
    return data_train, data_val, data_test
    
def get_my_dataset_dicts(json_file, category_dict, image_dir):
    """
        Convert a Label Studio json to COCO dataset json
    """
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)

    dataset_dicts = []
    for idx, image in enumerate(data['images']):
        record = {}
        # Create file path from local directory and image id
        filename = os.path.join(image_dir, image['id'] + '.jpg')

        # Assign image properties
        record["file_name"] = filename
        record["height"] = image["height"] 
        record["width"] = image["width"]
        record["image_id"] = image['id']

        annotations = []
        # Assign annotations to image
        for ann in data["annotations"]:
            for result in ann["result"]:
                if ann["id"] == image["id"] and type(result["label"]) is not int and result["label"][0] in category_dict.keys():
                # Create a new dict for each annotation in the image
                    x = int(result['bbox']['x'] / 100.0 * image["width"])
                    y = int(result['bbox']['y'] / 100.0 * image["height"])
                    width = int(result['bbox']['width'] / 100.0 * image["width"])
                    height = int(result['bbox']['height'] / 100.0 * image["height"])
                    obj = {
                        "bbox": [x, y, width, height],
                        "bbox_mode": BoxMode.XYWH_ABS,  # as your bounding box coordinates are in absolute format
                        "category_id": category_dict[result["label"][0]],  # map your label name to its corresponding id
                    }
                    annotations.append(obj)
        record["annotations"] = annotations
        dataset_dicts.append(record)
    return dataset_dicts
    
