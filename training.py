import sys
import time
import signal
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from logger import log
from model import Model3DRLDSRN
from plotting import generate_images
from loss_functions import supervised_loss, psnr_and_ssim_loss
from data_preparing import get_batch_data, data_pre_processing


def signal_handler(sig, frame):
    stop_log = "The training process was stopped at {}".format(time.ctime())
    log(stop_log)
    m.save_models("stopping_epoch")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def training_loop(N_TRAINING_DATA, lr_train, hr_train, BATCH_SIZE, epoch):
    for i in range(0, N_TRAINING_DATA, BATCH_SIZE):
        r = np.random.randint(0,2,3)
        batch_data = get_batch_data(lr_train, i, BATCH_SIZE, r[0], r[1], r[2])
        batch_label = get_batch_data(hr_train, i, BATCH_SIZE, r[0], r[1], r[2])
        m.training(batch_data, batch_label, epoch)

def evaluation_loop(N_DATA, lr_data, hr_data, PATCH_SIZE, BATCH_SIZE):
    hr_data, lr_data = shuffle(hr_data, lr_data)
    if N_DATA%BATCH_SIZE != 0:
        N_DATA = N_DATA - N_DATA%BATCH_SIZE
    output_data = np.empty((0,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1), 'float64')
    for i in range(0, N_DATA, BATCH_SIZE):        
        batch_data = get_batch_data(lr_data, i, BATCH_SIZE)
        output = m.generator_g(batch_data, training=False).numpy()
        output_data = np.append(output_data, output , axis=0)
    output_data = tf.squeeze(output_data).numpy()
    hr_data     = tf.squeeze(hr_data).numpy()[0:N_DATA]
    errors = []
    psnr  = tf.image.psnr(output_data, hr_data, 1)
    psnr  = psnr[psnr!=float("inf")]
    ssim  = tf.image.ssim(output_data, hr_data, 1)
    for v in [psnr, ssim]:
        v = tf.reduce_mean(v).numpy()
        v  = round(v,3)
        errors.append(v)
    errors.append(round(supervised_loss(output_data, hr_data).numpy(),3))
    return errors


def generate_random_image_slice(N_DATA, lr_data, hr_data, PATCH_SIZE, str1, str2=""):
    r_v = np.random.randint(0,N_DATA-1)
    comparison_image_lr = lr_data[r_v:r_v+1]
    prediction_image    = tf.squeeze(m.generator_g(comparison_image_lr, training=False)).numpy()
    comparison_image_lr = tf.squeeze(comparison_image_lr).numpy()
    comparison_image_hr = tf.squeeze(hr_data[r_v:r_v+1]).numpy()
    generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, str1, str2)


def main_loop(LR, EPOCHS, BATCH_SIZE, EPOCH_START, LAMBDA_ADV, LAMBDA_GRD_PEN,
              LAMBDA_CYC, LAMBDA_IDT, CRIT_ITER, TRAIN_ONLY, MODEL):

    begin_log = '\n### Began training {} at {} with parameters: Starting Epoch={}, Epochs={}, Batch Size={}, Learning Rate={}, Lambda Adversarial Loss={}, Lambda Cycle Loss={}, Lambda Identity Loss={}, Lambda Gradient Penalty={}, Critic iterations={}, Training only={}\n'.format(MODEL, time.ctime(),
     EPOCH_START, EPOCHS, BATCH_SIZE, LR, LAMBDA_ADV, LAMBDA_CYC, LAMBDA_IDT, LAMBDA_GRD_PEN, CRIT_ITER, TRAIN_ONLY)
    
    log(begin_log)

    training_start = time.time()

    lr_data = np.load('data/3d_lr_data.npy') # Low Resolution Data, Shape  : (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    hr_data = np.load('data/3d_hr_data.npy') # High Resolution Data, Shape : (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    
    PATCH_SIZES = hr_data.shape[1:4]         # (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    assert(PATCH_SIZES==lr_data.shape[1:4])
    PATCH_SIZE  = PATCH_SIZES[0]
    assert(PATCH_SIZE==PATCH_SIZES[1]==PATCH_SIZES[2])
    log("Patch Size is "+str(PATCH_SIZE))

    (hr_train, lr_train),(hr_validation, lr_validation),(hr_test, lr_test) = data_pre_processing(lr_data, hr_data, BATCH_SIZE)

    N_TRAINING_DATA, N_VALIDATION_DATA, N_TESTING_DATA = lr_train.shape[0], lr_validation.shape[0], lr_test.shape[0]
    
    nu_data_log = "Number of Training Data: {}, Number of Validation Data: {}, Number of Testing Data: {}".format(N_TRAINING_DATA, N_VALIDATION_DATA, N_TESTING_DATA)
    log(nu_data_log)
         
    global m
    m = Model3DRLDSRN(PATCH_SIZE=PATCH_SIZE, BATCH_SIZE=BATCH_SIZE, LR_G=LR, LR_D=LR, LAMBDA_ADV=LAMBDA_ADV,
     LAMBDA_GRD_PEN=LAMBDA_GRD_PEN, LAMBDA_CYC=LAMBDA_CYC, LAMBDA_IDT=LAMBDA_IDT, MODEL=MODEL, CRIT_ITER=CRIT_ITER,
     TRAIN_ONLY=TRAIN_ONLY)
        
    #Initial Random Slice Image Generation
    generate_random_image_slice(N_VALIDATION_DATA, lr_validation, hr_validation, PATCH_SIZE, 'a_first_plot_{}'.format(EPOCH_START))
    va_psnr, va_ssim, va_error = evaluation_loop(N_VALIDATION_DATA//60, lr_validation, hr_validation, PATCH_SIZE, BATCH_SIZE)
    
    evaluation_log = "Before training: Error = "+str(va_error)+", PSNR = "+str(va_psnr)+", SSIM = "+str(va_ssim)
    log(evaluation_log)    

    for epoch in range(EPOCH_START, EPOCH_START+EPOCHS):

        log("Began epoch {} at {}".format(epoch, time.ctime()))

        epoch_start = time.time()
        
        #TRAINING
        try:
            training_loop(N_TRAINING_DATA, lr_train, hr_train, BATCH_SIZE, epoch)
        except tf.errors.ResourceExhaustedError:
            log("Encountered OOM Error at {} !".format(time.ctime()))
            m.save_models(epoch)
            return

        tr_psnr, tr_ssim, generator_loss = evaluation_loop(N_TRAINING_DATA//100, lr_train, hr_train, PATCH_SIZE, BATCH_SIZE)
        
        hr_train, lr_train = shuffle(hr_train, lr_train)
        
        #Validation
        generate_random_image_slice(N_VALIDATION_DATA, lr_validation, hr_validation, PATCH_SIZE, "epoch_{}".format(epoch), str2=" Epoch: {}".format(epoch))
        va_psnr, va_ssim, va_error = evaluation_loop(N_VALIDATION_DATA//60, lr_validation, hr_validation, PATCH_SIZE, BATCH_SIZE)
        
        #Epoch Logging
        log("Finished epoch {} at {}.".format(epoch, time.ctime()))
        epoch_seconds = time.time() - epoch_start
        epoch_t_log = "Epoch took {}".format(datetime.timedelta(seconds=epoch_seconds))
        log(epoch_t_log)
        evaluation_log = "After epoch: Error = "+str(va_error)+", PSNR = "+str(va_psnr)+", SSIM = "+str(va_ssim)
        log(evaluation_log)

        if (epoch + 1) % 30 == 0:
            m.save_models(epoch)
            
    m.save_models("last_epoch")

    #Testing
    generate_random_image_slice(N_TESTING_DATA, lr_test, hr_test, PATCH_SIZE, 'z_testing_plot_{}'.format(EPOCH_START))
    test_psnr, test_ssim, test_error = evaluation_loop(N_TESTING_DATA, lr_test, hr_test, PATCH_SIZE, BATCH_SIZE)
        
    #Training Cycle Meta Data Logging
    evaluation_log = "After training: Error = "+str(test_error)+", PSNR = "+str(test_psnr)+", SSIM = "+str(test_ssim)
    log(evaluation_log)
    log("Finished training at {}".format(time.ctime()))
    training_seconds = time.time() - training_start
    training_t_log = "Training took {}".format(datetime.timedelta(seconds=training_seconds))
    log(training_t_log)
