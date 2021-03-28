import os
import sys
import time
import signal
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from logger import log
from model import Generator
from data_preparing import get_batch_data, fix_shape
from plotting import generate_images, plot_evaluations
from model_checkpoints import get_generator, save_generator
from loss_functions import supervised_loss, psnr_and_ssim_loss


def signal_handler(sig, frame):
    stop_log = "The training process was stopped at {}".format(time.ctime())
    log(stop_log)
    plot_evaluations(epochs_plot, training_psnr_plot, validation_psnr_plot, training_ssim_plot
                     , validation_ssim_plot, training_generator_g_error_plot,
                     validation_generator_g_error_plot, "stopped")
    save_generator(ckpt_manager, "stopping_epoch")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


@tf.function
def train_step(real_x, real_y):
    
    with tf.GradientTape(persistent=True) as tape:
        
        fake_y = generator_g(real_x, training=True)

        gen_g_super_loss = supervised_loss(real_y, fake_y)
                
    gradients_of_generator = tape.gradient(gen_g_super_loss, generator_g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_g.trainable_variables))
    
    return gen_g_super_loss


def training_loop(lr_train, hr_train, N_TRAINING_DATA, BATCH_SIZE):
    for i in range(0, N_TRAINING_DATA, BATCH_SIZE):
        r = np.random.randint(0,2,3)
        batch_data = get_batch_data(lr_train, i, BATCH_SIZE, r[0], r[1], r[2])
        batch_label = get_batch_data(hr_train, i, BATCH_SIZE, r[0], r[1], r[2])
        generator_loss = train_step(batch_data, batch_label).numpy()
    return generator_loss


def main_loop(LR_G, EPOCHS, BATCH_SIZE, LOSS_FUNC, EPOCH_START, NO_OF_DENSE_BLOCKS,
                  K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS, WEIGHTS_INIT):

    begin_log = '\n### Began training at {} with parameters: Starting Epoch={}, Epochs={}, Batch Size={}, Learning Rate={}, Loss Function={}, No. of Dense Blocks={}, Growth Rate(K)={}, No. of units per Block={}, Utilize Bias={}, Weights Initializer={} \n'.format(time.ctime(),
     EPOCH_START, EPOCHS, BATCH_SIZE, LR_G, LOSS_FUNC, NO_OF_DENSE_BLOCKS, K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS, WEIGHTS_INIT)
    
    log(begin_log)

    training_start = time.time()

    lr_data = np.load('data/3d_lr_data.npy') # (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    hr_data = np.load('data/3d_hr_data.npy') # (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    
    PATCH_SIZES = hr_data.shape[1:4]         # (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    assert(PATCH_SIZES==lr_data.shape[1:4])
    PATCH_SIZE  = PATCH_SIZES[0]
    assert(PATCH_SIZE==PATCH_SIZES[1]==PATCH_SIZES[2])

    
    lr_train, lr_temp, hr_train, hr_temp = train_test_split(lr_data, hr_data, test_size=1-0.1, random_state=42)
    lr_validation, lr_test, hr_validation, hr_test = train_test_split(lr_temp, hr_temp, test_size=0.1, random_state=42)
    
    N_TRAINING_DATA   = lr_train.shape[0]
    N_VALIDATION_DATA = lr_validation.shape[0]
    N_TESTING_DATA    = lr_test.shape[0]
    
    if N_TRAINING_DATA%BATCH_SIZE != 0:
        N_TRAINING_DATA = N_TRAINING_DATA - N_TRAINING_DATA%BATCH_SIZE
        lr_train = lr_train[0:N_TRAINING_DATA]
        hr_train = hr_train[0:N_TRAINING_DATA]
        
    no_data_log = "Number of Training Data: {}, Number of Validation Data: {}, Number of Testing Data: {}".format(N_TRAINING_DATA, N_VALIDATION_DATA, N_TESTING_DATA)
    log(no_data_log)
    
    patch_log = "Patch Size is "+str(PATCH_SIZE)
    log(patch_log)

    global generator_g, generator_optimizer, ckpt_manager
    generator_g, generator_optimizer, ckpt_manager = get_generator(PATCH_SIZE, LR_G, NO_OF_DENSE_BLOCKS, K,
     NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS, WEIGHTS_INIT)
    
    r_v = np.random.randint(0,N_VALIDATION_DATA)
    comparison_image_hr = hr_validation[r_v]
    comparison_image_lr = lr_validation[r_v]
    prediction_image = generator_g(comparison_image_lr, training=False).numpy()

    psnr, ssim = generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, 'a_first_plot_{}'.format(EPOCH_START))
    evaluation_log = "Before training: PSNR = "+str(psnr)+", SSIM = "+str(ssim)
    log(evaluation_log)

    global epochs_plot, training_generator_g_error_plot, training_psnr_plot, training_ssim_plot, validation_generator_g_error_plot, validation_psnr_plot, validation_ssim_plot
    epochs_plot = []
    training_generator_g_error_plot = []
    training_psnr_plot = []
    training_ssim_plot = []
    validation_generator_g_error_plot = []
    validation_psnr_plot = []
    validation_ssim_plot = []

    for epoch in range(EPOCH_START, EPOCH_START+EPOCHS):

        epoch_s_log = "Began epoch "+str(epoch)+" at "+time.ctime()
        log(epoch_s_log)

        epoch_start = time.time()
        
        #TRAINING
        generator_loss = training_loop(lr_train, hr_train, N_TRAINING_DATA, BATCH_SIZE)
                
        r_t = np.random.randint(0,N_TRAINING_DATA)
        comparison_image_hr = lr_train[r_t]
        comparison_image_lr = hr_train[r_t]
        prediction_image = generator_g(comparison_image_lr, training=False).numpy()
        tr_psnr, tr_ssim = psnr_and_ssim_loss(prediction_image,comparison_image_hr,PATCH_SIZE)
        
        training_generator_g_error_plot.append(generator_loss)
        training_psnr_plot.append(tr_psnr)
        training_ssim_plot.append(tr_ssim)
        
        hr_train, lr_train = shuffle(hr_train, lr_train)
        
        #Validation
        r_v = np.random.randint(0,N_VALIDATION_DATA)
        comparison_image_hr = hr_validation[r_v]
        comparison_image_lr = lr_validation[r_v]
        prediction_image    = generator_g(comparison_image_lr, training=False).numpy()
        va_psnr, va_ssim = generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, 'a_first_plot_{}'.format(EPOCH_START))
        va_error         = supervised_loss(fix_shape(prediction_image,PATCH_SIZE), fix_shape(comparison_image_hr,PATCH_SIZE)).numpy()
        
        validation_psnr_plot.append(va_psnr)
        validation_ssim_plot.append(va_ssim)
        validation_generator_g_error_plot.append(va_error)
        
        #Epoch Logging
        epochs_plot.append(epoch)
        epoch_e_log = "Finished epoch {} at {}.".format(epoch, time.ctime())
        log(epoch_e_log)
        epoch_seconds = time.time() - epoch_start
        epoch_t_log = "Epoch took {}".format(datetime.timedelta(seconds=epoch_seconds))
        log(epoch_t_log)
        evaluation_log = "After epoch: Error = {}, PSNR = {}, SSIM = {}".format(va_error, va_psnr,va_ssim)
        log(evaluation_log)

        if (epoch + 1) % 50 == 0:
            save_generator(ckpt_manager, epoch)
            plot_evaluations(epochs_plot, training_psnr_plot, validation_psnr_plot, training_ssim_plot
                             , validation_ssim_plot, training_generator_g_error_plot,
                             validation_generator_g_error_plot, EPOCH_START)
            
    plot_evaluations(epochs_plot, training_psnr_plot, validation_psnr_plot, training_ssim_plot
                     , validation_ssim_plot, training_generator_g_error_plot,
                     validation_generator_g_error_plot, EPOCH_START)

    #Testing
    r_v = np.random.randint(0,N_TESTING_DATA)
    comparison_image_hr  = hr_test[r_v]
    comparison_image_lr  = lr_test[r_v]
    prediction_image     = generator_g(comparison_image_lr, training=False).numpy()
    test_psnr, test_ssim = generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, 'z_testing_plot_{}'.format(EPOCH_START))
    test_error           = supervised_loss(fix_shape(prediction_image,PATCH_SIZE), fix_shape(comparison_image_hr,PATCH_SIZE)).numpy()

    evaluation_log = "After training: Error = {}, PSNR = {}, SSIM = {}".format(test_error, test_psnr, test_ssim)
    log(evaluation_log)
    
    training_e_log = "Finished training at {}".format(time.ctime())
    log(training_e_log)
    training_seconds = time.time() - training_start
    training_t_log = "Training took {}".format(datetime.timedelta(seconds=training_seconds))
    log(training_t_log)
