import gradio as gr
from PIL import Image
import zipfile
import io
import os

# Function to list files in the current directory
def list_files_in_directory():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    return files

# Processing function for the uploaded file or selected file
def process_input(file_path_or_uploaded_file, from_dropdown):
    image_paths = []

    # Function to process each file within a ZIP, supporting nested directories
    def process_zip(zip_ref, path=""):
        for zip_info in zip_ref.infolist():
            if zip_info.is_dir():
                process_zip(zip_ref, zip_info.filename)  # Recursive call for directories
            elif zip_info.filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(zip_info.filename)
                with zip_ref.open(zip_info) as image_file:
                    image = Image.open(image_file)
                    image.load()  # Necessary for PIL to process image data
                    return image  # Returns the first image found

    if from_dropdown:  # If the file is selected from the dropdown
        if zipfile.is_zipfile(file_path_or_uploaded_file):
            with zipfile.ZipFile(file_path_or_uploaded_file, 'r') as zip_ref:
                return process_zip(zip_ref)
        else:
            return Image.open(file_path_or_uploaded_file)
    else:  # If the file is uploaded
        if zipfile.is_zipfile(file_path_or_uploaded_file):
            with zipfile.ZipFile(file_path_or_uploaded_file, 'r') as zip_ref:
                return process_zip(zip_ref)
        else:
            return Image.open(file_path_or_uploaded_file)

files_dropdown = gr.Dropdown(label="Select a file from current directory", choices=list_files_in_directory())
file_upload = gr.File(label="Upload a ZIP file or an Image")

interface = gr.Interface(fn=process_input,
                         inputs=[gr.Radio(choices=["Upload", "Select from directory"], label="Input Method"), file_upload, files_dropdown],
                         outputs=gr.Image(label="Processed Image"),
                         examples=[])

interface.launch()