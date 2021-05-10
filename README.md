# 3DRLD-SRN
## Introduction
This project is a work in progress that utilizes deep CNNs to perform super resolution on 3D models from CT/MRI scans.

## Data Format Requirements
The image scans have to be in the 'nii' format and be placed in the 'data' directory where the program will recursively search through all folders there to gather all the images available.

## Training
Use the following command to check possible input arguements for training.
```
python main.py -h 
```