# 3DRLD-SRN
## Introduction
This project is a work in progress that utilizes deep CNNs to perform super resolution on 3D models from CT/MRI scans.

## Data Format Requirements
The training data has to be in split into two files in the data directory: (3d_lr_data.npy) containing the low resolution patches and (3d_hr_data.npy) containing their equivalent high resolution patches.

Both should have the numpy shape (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1) where N is the number of patches available.

## Training
Use the following command to check possible input arguements for training.
```
python main.py -h 
```