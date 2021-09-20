# 3DRDN
## Introduction
This project is a work in progress that utilizes deep CNNs to perform super resolution on 3D models from CT/MRI scans.

## Data Format Requirements
The image scans have to be in the 'nii' format and be placed in the 'data' directory where the program will recursively search through all folders there to gather all the images available.

## Training

Use the command line to begin training on the dataset and select the model architecture as well as the training parameters like in the following example.
```
python main.py --model CYCLE-WGANGP-3DRLDSRN --learning_rate 5e-5 --batch_size 2 --lambda_adv 0.02 --epochs 100
```

Use the following command to check possible input arguements for training.
```
python main.py -h 
```