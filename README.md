Author : Mayukha Thumiki
Programming Language Used: Python
Python Version: 3.9.13
Platform: Jupyter Notebooks (Google Colab)
GPU Requirement: T4 GPU

Structure
This project implements a Truck Detection System using a custom-trained YOLOv5 model. The system is designed to detect trucks in images and classify other vehicles as "Other Vehicle." The pipeline includes setting up the environment, preprocessing datasets, training a custom model, and evaluating its performance.

Setup
Environment Setup:
Clone the YOLOv5 repository and install required packages:
    	!git clone https://github.com/ultralytics/yolov5.git
    	!cd yolov5 && pip install -r requirements.txt
    	!pip install roboflow
    	!pip install opendatasets


Model Setup
Load pre-trained YOLOv5 models and customize the architecture by incorporating a Transformer block for enhanced feature extraction.

Dataset Preparation
Convert datasets into YOLO format and merge datasets for training, validation, and testing. The script processes the datasets by converting annotations and saving them in YOLO format.

Training
The model is trained using a custom configuration file (yolov5s_config.yaml) and hyperparameters (hyp.yaml). The training script also allows for freezing layers to focus training on specific parts of the network.

Testing and Evaluation
The system evaluates the model on test data, generating performance metrics such as Precision, Recall, mAP, and confusion matrices. The results are visualized through graphs like PR curves and confusion matrices.

Inference
The trained model can be used for inference on new images, including images fetched from an API, displaying the results with bounding boxes and classifications.

Key Functions
- convert_to_yolo_format: Converts bounding box coordinates to YOLO format.
- process_directory: Processes datasets and converts annotations to YOLO format.
- count_labels: Counts the number of truck and non-truck instances in label files.
- plot_box: Visualizes bounding boxes on images.
- TransformerBlock & C3TR: Custom layers incorporating Transformers into YOLOv5.

Usage
- Ensure all paths are correctly set up for the images and labels.
- Modify parameters as necessary for custom configurations and training runs.
- Run the training script to train the model and the validation script to evaluate it.

Additional Notes
- Hyperparameters, class mappings, and model architecture can be adjusted in the configuration files.
- Visualization of results helps in analyzing the performance of the model and making necessary adjustments.
