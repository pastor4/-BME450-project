# Title
"Early Detection of Brain Tumors Using MRI and Deep Learning"

## Team Members
Anthony Pastor (pastor4)

## Project Description
I aim to build a deep learning model to classify brain tumors from T1-weighted contrast-enhanced MRI images. The model will be trained to distinguish between three common tumor types: glioma, meningioma, and pituitary tumors, as well as scans with no tumors present. I plan on using publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which contains over 3,000 MRI images labeled by tumor type, already classified into testing and training sets. The goal is to preprocess the images, train a CNN model using PyTorch, and evaluate performance with metrics such as accuracy, precision, and recall. The model should be able to accurately screen an individual for a brain tumor and determine which (if any) type of brain tumor is detected. If there is time, I'd like to try and map where specifically on the MRI scan the model found each tumor (or where it falsly labeled one).
