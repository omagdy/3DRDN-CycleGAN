import sys
import time
import signal
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from logger import log
from model import Model3DRLDSRN
from plotting import generate_images
from loss_functions import supervised_loss, psnr_and_ssim_loss
from data_preprocessing import get_preprocessed_data, PATCH_SIZE


def signal_handler(sig, frame):
    stop_log = "The training process was stopped at {}".format(time.ctime())
    log(stop_log)
    m.save_models("stopping_epoch")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def evaluation_loop(dataset, PATCH_SIZE):
    output_data = np.empty((0,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1), 'float64')
    hr_data     = np.empty((0,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1), 'float64')
    for lr_image, hr_image in dataset:        
        output = m.generator_g(lr_image, training=False).numpy()
        output_data = np.append(output_data, output , axis=0)
        hr_data     = np.append(hr_data, hr_image , axis=0)
    output_data = tf.squeeze(output_data).numpy()
    hr_data     = tf.squeeze(hr_data).numpy()
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


def generate_random_image_slice(sample_image, PATCH_SIZE, str1, str2=""):
    comparison_image_lr, comparison_image_hr = sample_image
    comparison_image_lr = tf.expand_dims(comparison_image_lr, axis=0)
    prediction_image    = tf.squeeze(m.generator_g(comparison_image_lr, training=False)).numpy()
    comparison_image_lr = tf.squeeze(comparison_image_lr).numpy()
    comparison_image_hr = tf.squeeze(comparison_image_hr).numpy()
    generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, str1, str2)


def main_loop(LR, DB, DU, EPOCHS, BATCH_SIZE, EPOCH_START, LAMBDA_ADV, LAMBDA_GRD_PEN,
              LAMBDA_CYC, LAMBDA_IDT, CRIT_ITER, TRAIN_ONLY, MODEL):

    begin_log = '\n### Began training {} at {} with parameters: Starting Epoch={}, Epochs={}, Dense Blocks={}, Dense Units={}, Batch Size={}, Learning Rate={}, Lambda Adversarial Loss={}, Lambda Cycle Loss={}, Lambda Identity Loss={}, Lambda Gradient Penalty={}, Critic iterations={}, Training only={}\n'.format(MODEL, time.ctime(),
     EPOCH_START, EPOCHS, DB, DU, BATCH_SIZE, LR, LAMBDA_ADV, LAMBDA_CYC, LAMBDA_IDT, LAMBDA_GRD_PEN, CRIT_ITER, TRAIN_ONLY)
    
    log(begin_log)

    training_start = time.time()

    log("Setting up Data Pipeline")
    VALIDATION_BATCH_SIZE = 5
    train_dataset, valid_dataset, test_dataset = get_preprocessed_data(BATCH_SIZE, VALIDATION_BATCH_SIZE)
    pipeline_seconds = time.time() - training_start
    pipeline_t_log = "Pipeline took {} to set up".format(datetime.timedelta(seconds=pipeline_seconds))
    log(pipeline_t_log)

    N_TRAINING_DATA   = train_dataset.cardinality().numpy()*BATCH_SIZE
    N_VALIDATION_DATA = valid_dataset.cardinality().numpy()*VALIDATION_BATCH_SIZE
    N_TESTING_DATA    = test_dataset.cardinality().numpy()*BATCH_SIZE
    
    nu_data_log = "Number of Training Data: {}, Number of Validation Data: {}, Number of Testing Data: {}".format(N_TRAINING_DATA, N_VALIDATION_DATA, N_TESTING_DATA)
    log(nu_data_log)

    valid_dataset = valid_dataset.repeat().prefetch(1).as_numpy_iterator()
         
    global m
    m = Model3DRLDSRN(PATCH_SIZE=PATCH_SIZE, DB=DB, DU=DU, BATCH_SIZE=BATCH_SIZE, LR_G=LR, LR_D=LR, LAMBDA_ADV=LAMBDA_ADV,
     LAMBDA_GRD_PEN=LAMBDA_GRD_PEN, LAMBDA_CYC=LAMBDA_CYC, LAMBDA_IDT=LAMBDA_IDT, MODEL=MODEL, CRIT_ITER=CRIT_ITER,
     TRAIN_ONLY=TRAIN_ONLY)
        
    #Initial Random Slice Image Generation
    valid_batch = [next(valid_dataset)]
    va_psnr, va_ssim, va_error = evaluation_loop(valid_batch, PATCH_SIZE)
    sample_image = (valid_batch[0][0][0], valid_batch[0][1][0])
    generate_random_image_slice(sample_image, PATCH_SIZE, 'a_first_plot_{}'.format(EPOCH_START), str2="")

    evaluation_log = "Before training: Error = "+str(va_error)+", PSNR = "+str(va_psnr)+", SSIM = "+str(va_ssim)
    log(evaluation_log)    

    for epoch in range(EPOCH_START, EPOCH_START+EPOCHS):

        log("Began epoch {} at {}".format(epoch, time.ctime()))

        epoch_start = time.time()
        
        #TRAINING
        try:
            for lr, hr in train_dataset:
                m.training(lr, hr, epoch)
        except tf.errors.ResourceExhaustedError:
            log("Encountered OOM Error at {} !".format(time.ctime()))
            m.save_models(epoch)
            return
                
        #Validation
        valid_batch = [next(valid_dataset)]
        va_psnr, va_ssim, va_error = evaluation_loop(valid_batch, PATCH_SIZE)
        sample_image = (valid_batch[0][0][0], valid_batch[0][1][0])
        generate_random_image_slice(sample_image, PATCH_SIZE, "epoch_{}".format(epoch), str2=" Epoch: {}".format(epoch))

        with m.summary_writer_valid.as_default():
            tf.summary.scalar('Mean Absolute Error', va_error, step=epoch)
            tf.summary.scalar('PSNR', va_psnr, step=epoch)
            tf.summary.scalar('SSIM', va_ssim, step=epoch)
        
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
    generate_random_image_slice(sample_image, PATCH_SIZE, 'z_testing_plot_{}'.format(EPOCH_START))
    test_psnr, test_ssim, test_error = evaluation_loop(test_dataset, PATCH_SIZE)
        
    #Training Cycle Meta Data Logging
    evaluation_log = "After training: Error = "+str(test_error)+", PSNR = "+str(test_psnr)+", SSIM = "+str(test_ssim)
    log(evaluation_log)
    log("Finished training at {}".format(time.ctime()))
    training_seconds = time.time() - training_start
    training_t_log = "Training took {}".format(datetime.timedelta(seconds=training_seconds))
    log(training_t_log)
