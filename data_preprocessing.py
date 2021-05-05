import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import tensorflow_addons as tfa
from numpy.random import randint


PATCH_SIZE           = 40
GAUSSIAN_NOISE       = 0.25
SCALING_FACTOR       = 2
interpolation_method = 'bicubic'


@tf.function
def NormalizeImage(image):
    return (image - tf.math.reduce_min(image)) / (tf.math.reduce_max(image) - tf.math.reduce_min(image))

def get_random_patch_dims(image):
    x, y, z = image.shape[0], image.shape[1], image.shape[2]
    threshold = 0.30
    x_thr, y_thr, z_thr = int(x*threshold), int(y*threshold), int(z*threshold)
    r_x = randint(x_thr, x-PATCH_SIZE-x_thr)
    r_y = randint(y_thr, y-PATCH_SIZE-y_thr)
    r_z = randint(z_thr, z-PATCH_SIZE-z_thr)
    return r_x, r_y, r_z

def flip_model_x(model):
    return model[::-1, :, :]

def flip_model_y(model):
    return model[:, ::-1, :]

def flip_model_z(model):
    return model[:, :, ::-1]

@tf.function
def data_augmentation(lr_image, hr_image):
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_x(lr_image), flip_model_x(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_y(lr_image), flip_model_y(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_z(lr_image), flip_model_z(hr_image)
    return lr_image, hr_image

def get_nii_file(nii_file_path):
    img_sitk = sitk.ReadImage(nii_file_path.decode('UTF-8'), sitk.sitkFloat64)
    hr_image = sitk.GetArrayFromImage(img_sitk)
    return hr_image

@tf.function
def add_noise(hr_image):
    blurred_image = tfa.image.gaussian_filter2d(hr_image, sigma=GAUSSIAN_NOISE)
    return blurred_image

def get_low_res(hr_image):
    x, y, z = hr_image.shape
    lr_image   = tf.image.resize(hr_image, [x//SCALING_FACTOR, y//SCALING_FACTOR], method=interpolation_method).numpy()
    lr_image = np.rot90(lr_image, axes=(1,2))
    lr_image = tf.image.resize(lr_image, [x//SCALING_FACTOR, z//SCALING_FACTOR], method=interpolation_method).numpy()
    ups_lr_image = tf.image.resize(lr_image, [x//SCALING_FACTOR, z], method=interpolation_method).numpy()
    ups_lr_image = np.rot90(ups_lr_image, axes=(1,2))
    ups_lr_image = tf.image.resize(ups_lr_image, [x, y], method=interpolation_method).numpy()
    ups_lr_image = np.array(np.rot90(ups_lr_image, k=2, axes=(1,2)), dtype='float64')
    return ups_lr_image, hr_image

@tf.function
def normalize(lr_image, hr_image):
    hr_image = NormalizeImage(hr_image)
    lr_image = NormalizeImage(lr_image)
    return lr_image, hr_image

def extract_patch(lr_image, hr_image):
    r_x, r_y, r_z = get_random_patch_dims(hr_image)
    hr_random_patch = hr_image[r_x:r_x+PATCH_SIZE,r_y:r_y+PATCH_SIZE,r_z:r_z+PATCH_SIZE]
    lr_random_patch = lr_image[r_x:r_x+PATCH_SIZE,r_y:r_y+PATCH_SIZE,r_z:r_z+PATCH_SIZE]
    return tf.expand_dims(lr_random_patch, axis=4), tf.expand_dims(hr_random_patch, axis=4)

def get_preprocessed_data(BATCH_SIZE):
    nii_files = glob.glob("data/**/*.nii", recursive=True)
    nii_files=np.array(nii_files)

    dataset  = tf.data.Dataset.from_tensor_slices(nii_files)
    AUTOTUNE = tf.data.AUTOTUNE
    dataset  = dataset.map( lambda x: tf.numpy_function(func=get_nii_file, inp=[x], Tout=tf.float64), 
                            num_parallel_calls=AUTOTUNE, deterministic=False)
    dataset  = dataset.map(add_noise, num_parallel_calls=AUTOTUNE, deterministic=False)
    dataset  = dataset.map( lambda x: tf.numpy_function(func=get_low_res, inp=[x], Tout=(tf.float64, tf.float64)),
                             num_parallel_calls=AUTOTUNE, deterministic=False)
    dataset  = dataset.map(normalize, num_parallel_calls=AUTOTUNE, deterministic=False)
    dataset  = dataset.map( lambda x,y: tf.numpy_function(func=extract_patch, inp=[x,y], Tout=(tf.float64, tf.float64)),
                             num_parallel_calls=AUTOTUNE, deterministic=False)
    
    sample_image  = dataset.take(30).batch(1).cache('cache/sample_image')
    for l,h in sample_image: # Iterating until all sample images are cached
        pass
    sample_image  = sample_image.shuffle(30).take(1).repeat().as_numpy_iterator()
    dataset  = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset_size = dataset.cardinality().numpy()
    train_data_threshold  = int(0.5*dataset_size) # 50% of the dataset
    train_dataset = dataset.take(train_data_threshold).prefetch(AUTOTUNE)

    remain_dataset = dataset.skip(train_data_threshold)
    remain_dataset_size = remain_dataset.cardinality().numpy()
    valid_data_threshold  = int(0.5*remain_dataset_size)
    valid_dataset = remain_dataset.take(valid_data_threshold).take(3).cache('cache/validation_data_cache')
    for l_batch,h_batch in valid_dataset: # Iterating until all validation data is cached
        pass

    test_dataset  = remain_dataset.skip(valid_data_threshold).take(10).prefetch(AUTOTUNE)

    return train_dataset, sample_image, valid_dataset, test_dataset
