# Flooded Road Detection with Deep Learning

This repository contains the implementation for detecting flooded and non-flooded roads using deep learning techniques applied on satellite images. The project utilizes various UNet-based architectures to process high-resolution satellite images and MODIS images for effective flood detection.

As the datasets of VHR is too huge, the datasets are excluded from this package. To acquire that, you can 1. contact me by my emial(chongyang_du@hust.edu.cn) or 2. download the dataset from the website of SpaceNet8.

This is my first repository on machine learning. If it does not work, feel free to ask me.

## Table of Contents
- [Flooded Road Detection with Deep Learning](#flooded-road-detection-with-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Results](#results)
  - [References](#references)

## Introduction
This project aims to accurately detect flooded roads using deep learning models applied on satellite images. The focus is on improving the emergency response capabilities during flood events by providing timely and accurate information about road conditions. The models are trained and evaluated on datasets combining high-resolution satellite images and MODIS images.

## Requirements
- Python 3.8+
- PyTorch
- GDAL
- NumPy
- Matplotlib
- Other dependencies specified in `requirements.txt`

## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/dcyd/flooded-road-detection.git
    cd flooded-road-detection
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation
1. Download and preprocess the satellite and MODIS images. Use `generate_train_val_test_csvs_sp8.py` to generate the CSV files that list the training, validation, and test images.

    ```bash
    python generate_train_val_test_csvs_sp8.py --output_dir ./data/csvs
    ```

2. Ensure the images are stored in the appropriate directory structure expected by the dataset loader.

## Training the Model
1. To train the model using high-resolution satellite images, use `train_flood_road_random.py`:

    ```bash
    python train_flood_road_random.py --train_csv ./data/csvs/train_data.csv --val_csv ./data/csvs/val_data.csv --save_dir ./model_output --model_name resnet18_siamese --lr 0.0001 --batch_size 2 --n_epochs 50 --gpu 0
    ```

2. To train the model using both high-resolution satellite images and MODIS images, use `train_modis_flood_road_random.py`:

    ```bash
    python train_modis_flood_road_random.py --train_csv ./data/csvs/train_data.csv --val_csv ./data/csvs/val_data.csv --save_dir ./model_output --model_name resnet18_siamese --lr 0.0001 --batch_size 2 --n_epochs 50 --gpu 0
    ```

## Evaluating the Model
1. To evaluate the trained model, use `flood_road_eval_random.py`:

    ```bash
    python flood_road_eval_random.py --model_path ./model_output/best_model.pth --model_name resnet18_siamese --in_csv ./data/csvs/test_data.csv --save_fig_dir ./results/pngs --save_preds_dir ./results/tifs --gpu 0
    ```

2. To evaluate the model trained with MODIS data, use `flood_road_modis_eval_random.py`:

    ```bash
    python flood_road_modis_eval_random.py --model_path ./model_output/best_model.pth --model_name resnet18_siamese --in_csv ./data/csvs/test_data.csv --save_fig_dir ./results/pngs --save_preds_dir ./results/tifs --gpu 0
    ```

## Results
The results are saved in the specified directories for figures and predictions. Evaluation metrics such as IoU, F1-score, precision, and recall are computed for both flooded and non-flooded roads.

## References
1. "Impact of Climate Change on Flooding and Infrastructure," Journal of Environmental Science.
2. "Economic Impact of Flooding on Road and Rail Infrastructure," World Bank Report.
3. "Deep Learning for Satellite Image Analysis," IEEE Transactions on Geoscience and Remote Sensing.
