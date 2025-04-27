# 367A Project - Detecting Deforestation Drivers
This repository contains our solutions for the "Identifying Deforestation Drivers - Solafune competition"

## Requirements

### Datasets

#### Solafune data
Download the datasets from solafune and organize them as this:

```
data/
├── evaluation_images/
│   ├── evaluation_0.tif
│   ├── evaluation_1.tif
│   ├── evaluation_2.tif
│   ├── ...
├── train_images/
│   ├── train_0.tif
│   ├── train_1.tif
│   ├── train_2.tif
│   ├── ...
├── train_annotations.json
```

#### DiffCR data
After preprocessing the images with DiffCR the data folder should look like this:

```
data/
├── evaluation_images/
│   ├── evaluation_0.tif
│   ├── evaluation_1.tif
│   ├── evaluation_2.tif
│   ├── ...
├── train_images/
│   ├── train_0.tif
│   ├── train_1.tif
│   ├── train_2.tif
│   ├── ...
├── evaluation_cleaned/
│   ├── evaluation_0.tif
│   ├── evaluation_1.tif
│   ├── evaluation_2.tif
│   ├── ...
├── train_cleaned/
│   ├── train_0.tif
│   ├── train_1.tif
│   ├── train_2.tif
│   ├── ...
├── train_annotations.json
```

### Libraries

Install the python packages imported the first cell of the notebooks.

## Usage

### Generate Masks
Use generate_masks.ipynb to generate segmentation masks for the training, and saves them as .npy files.

### Preprocess images with DiffCR
Use the provided preprocessing script inside baseline_&_DiffCR.py to remove clouds from the training and evaluation images by applying a pretrained DiffCR model patch-wise and saving the cleaned outputs.
This script creates two new folders: patch-wise and saving the cleaned outputs. This script creates two new folders, evaluation_images -> evaluation_cleaned and train_images -> train_cleaned.


### Train models
#### Train DiffCR Model
Navigate to external/diffcr_custom/ to train the DiffCR model.
Run the run.py script to start training; it will save the model weights as .pth files for each epoch in the checkpoints folder.

#### Train Baseline Model
train_model_baseline.ipynb contains the code for training our baseline model, it will save the model as a ckpt file and create a json file for submitting on solafune.

#### Train DiffCR + Baseline Model
baseline_&_diffcr.ipynb contains the code for training the baseline model with the DiffCR preprocessed images (cleaned images), it will save the model as a ckpt file and create a json file for submitting on solafune.

#### Train R2U++ Model
R2Upp_50epochs.ipynb contains the code for training our R2U++ model, it will save the model as a ckpt file and create a json file for submitting on solafune.

#### Train DiffCR + R2U++ Model
R2Upp_DiffCR_50epochs.ipynb contains the code for training the R2U++ model with the DiffCR preprocessed images (cleaned images), it will save the model as a ckpt file and create a json file for submitting on solafune.

