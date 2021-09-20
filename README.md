# 3DRDN-CGAN
## Introduction
This repository contains the implementation of my master thesis "[Deep Learning for 3D Super-Resolution](https://www.researchgate.net/publication/354708583_Deep_Learning_for_3D_Super-Resolution)" in which a 3D CNN integrated in a Cycle-GAN architecture is used to perform super resolution on 3D models from CT/MRI scans.

## Data Format Requirements
The image scans have to be in the 'nii' format and be placed in the 'data' directory where the program will recursively search through all folders there to gather all the images available.

## Training

Use the command line to begin training on the dataset and select the model architecture as well as the training parameters like in the following example.
```
python main.py --model CYCLE-WGANGP-3DRLDSRN --learning_rate 5e-5 --batch_size 2 --lambda_adv 0.02 --epochs 100
```
The trained model parameters will be saved in the tensor_checkpoints directory after the training is done.

The following command can be used to check all possible input arguements and hyperparameters.
```
python main.py -h 
```

## Contact 
If you have any questions, please file an issue or reach me via email:
```
Omar Hussein: omagdy.222@gmail.com
```

If my work is useful for your research, please consider citing it:

```shell
@article{article,
author = {Hussein, Omar and Sauer, Tomas},
year = {2021},
month = {07},
title = {Deep Learning for 3D Super-Resolution},
doi = {10.13140/RG.2.2.23152.58884}
}
```
